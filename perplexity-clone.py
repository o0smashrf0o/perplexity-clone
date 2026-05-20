#!/usr/bin/env python3
"""
local_perplexity_cli.py — Simple single-file local Perplexity-style assistant.

Features (all local):
- Ollama LLM via Agno
- Web search via SearXNG
- Web crawling via Crawl4AI
- File tools (read/write)
- Shell tools (safe, read-only)
- Tiny Python runner for calculations

This is a minimal architecture prototype, not production code.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import httpx
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools import tool
from agno.tools.crawl4ai import Crawl4aiTools
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools

# ---------------------------------------------------------------------
# Config + logging
# ---------------------------------------------------------------------
@dataclass
class Config:
    model_id: str = "gemma4:latest"
    searxng_url: str = "http://127.0.0.1:8080"
    max_search_results: int = 5
    max_page_length: int = 6000
    crawl_timeout: int = 30
    max_tool_calls: int = 20
    run_timeout: int = 300
    max_history_turns: int = 8
    workspace_dir: Path = Path(".").resolve()


CFG = Config()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("local_perplexity")


# ---------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------
history: deque[Dict[str, str]] = deque(maxlen=CFG.max_history_turns)


def build_prompt(user_message: str) -> str:
    if not history:
        return user_message
    lines: List[str] = []
    for i, turn in enumerate(history, start=1):
        lines.append(f"Turn {i} - User: {turn['user']}")
        lines.append(f"Turn {i} - Assistant: {turn['assistant']}")
    history_block = "\n".join(lines)
    return (
        "Use the recent conversation history below to answer the next user message.\n\n"
        f"Recent conversation history:\n{history_block}\n\n"
        f"Current user message:\n{user_message}\n"
    )


# ---------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------
def make_searxng_tool(searxng_url: str, max_results: int):
    @tool(
        name="searxng_search",
        description=(
            "Search the web using a local SearXNG instance. "
            "Input: a search query string. Output: top results with titles, URLs, and snippets."
        ),
    )
    def searxng_search(query: str) -> str:
        log.info("searxng_search → %r", query)
        endpoint = f"{searxng_url.rstrip('/')}/search"
        try:
            resp = httpx.get(endpoint, params={"q": query, "format": "json"}, timeout=10)
            resp.raise_for_status()
            results = resp.json().get("results", [])
            if not results:
                return "No results found."
            lines: List[str] = []
            for r in results[:max_results]:
                lines.append(
                    f"Title: {r.get('title')}\n"
                    f"URL: {r.get('url')}\n"
                    f"Snippet: {r.get('content')}\n---"
                )
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            log.warning("  search error: %s", exc)
            return f"Search error: {exc}"

    return searxng_search


def make_batch_crawl_tool(max_length: int, timeout: int):
    @tool(
        name="batch_web_crawler",
        description=(
            "Crawl several URLs at once and return each page's content. "
            "Input: comma-separated URLs. Output: mapping URL → text."
        ),
    )
    async def batch_web_crawler(urls: str) -> Dict[str, str]:
        try:
            from crawl4ai import AsyncWebCrawler  # type: ignore
        except ImportError:
            return {"error": "crawl4ai not installed"}

        url_list = [u.strip() for u in urls.split(",") if u.strip()]
        log.info("batch_web_crawler → %d URL(s)", len(url_list))

        async def _fetch(url: str) -> tuple[str, str]:
            try:
                async with AsyncWebCrawler(verbose=False) as crawler:
                    result = await asyncio.wait_for(
                        crawler.arun(url=url),
                        timeout=timeout,
                    )
                    text = getattr(result, "markdown", None) or getattr(
                        result, "extracted_content", ""
                    ) or ""
                    return url, text[:max_length]
            except asyncio.TimeoutError:
                return url, f"[Timeout after {timeout}s]"
            except Exception as exc:  # noqa: BLE001
                return url, f"[Error: {exc}]"

        pairs = await asyncio.gather(*[_fetch(u) for u in url_list])
        return dict(pairs)

    return batch_web_crawler


@tool(
    name="python_runner",
    description=(
        "Run short Python code snippets for calculations or tiny analyses. "
        "Input: Python code as a string. Output: stdout and/or errors."
    ),
)
def python_runner(code: str) -> str:
    log.info("python_runner called")
    banned = ["import os", "import subprocess", "open(", "rm -rf", "sys.exit"]
    if any(b in code for b in banned):
        return "Refused: potentially unsafe Python code."
    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        out = result.stdout.strip()
        err = result.stderr.strip()
        if err:
            return f"STDOUT:\n{out}\n\nSTDERR:\n{err}"
        return out or "(no output)"
    except Exception as exc:  # noqa: BLE001
        return f"Error running Python: {exc}"


# ---------------------------------------------------------------------
# Instructions + sanitization
# ---------------------------------------------------------------------
DESCRIPTION = (
    "You are a local Perplexity-style assistant. "
    "You can search the web, crawl pages, inspect local files, run safe shell commands, "
    "and execute small Python snippets for calculations."
)


def build_instructions(cfg: Config) -> List[str]:
    return [
        f"Current date/time: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "You run locally and have tools.",
        "Use web search (searxng_search) and Crawl4AI for questions requiring current web data.",
        "Use FileTools when the user explicitly mentions reading or writing workspace files.",
        "Use ShellTools only for safe, read-only commands inside the workspace (ls, cat, grep).",
        "Never run destructive shell commands (rm, sudo, chmod -R, kill, etc.).",
        "Keep all file and shell operations inside the workspace directory.",
        "Use python_runner only for short calculations or simple analysis.",
        "When you use web content, always include inline citations like [Source](URL).",
        "Keep answers concise and in Markdown.",

        "First, briefly plan which tools you need.",
        "Then discover URLs if needed, crawl pages, and synthesize an answer with sources.",
    ]


def sanitize_output(text: str) -> str:
    cleaned = text.strip()
    if "<channel|>" in cleaned:
        cleaned = cleaned.split("<channel|>")[-1].strip()
    phase_block = re.compile(
        r"^## PHASE\s+\d+.*?(?=^## PHASE\s+\d+|\Z)",
        re.MULTILINE | re.DOTALL,
    )
    cleaned = phase_block.sub("", cleaned).strip()
    return cleaned or text.strip()


# ---------------------------------------------------------------------
# Build agent
# ---------------------------------------------------------------------
def build_agent(cfg: Config) -> Agent:
    tools: List[Any] = [
        make_searxng_tool(cfg.searxng_url, cfg.max_search_results),
        Crawl4aiTools(max_length=cfg.max_page_length),
        make_batch_crawl_tool(cfg.max_page_length, cfg.crawl_timeout),
        ShellTools(),
        FileTools(),
        python_runner,
    ]
    agent = Agent(
        model=Ollama(id=cfg.model_id),
        tools=tools,
        description=DESCRIPTION,
        instructions=build_instructions(cfg),
        tool_call_limit=cfg.max_tool_calls,
        debug_mode=True,
        markdown=True,
    )
    log.info("Agent ready | model=%s | searxng=%s", cfg.model_id, cfg.searxng_url)
    return agent


# ---------------------------------------------------------------------
# CLI chat loop (no API)
# ---------------------------------------------------------------------
async def chat_loop() -> None:
    agent = build_agent(CFG)
    print("Local Perplexity-style agent ready. Type 'exit' or 'quit' to stop.\n")

    while True:
        user = input("You: ").strip()
        if not user:
            continue
        if user.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        prompt = build_prompt(user)
        try:
            resp = await agent.arun(prompt)
            answer = sanitize_output(resp.content or "")
        except Exception as exc:  # noqa: BLE001
            log.exception("Agent error")
            answer = f"Error from agent: {exc}"

        print(f"\nAssistant:\n{answer}\n")
        history.append({"user": user, "assistant": answer})

        if getattr(resp, "tools", None):
            print("Tools used:", [t.tool_name for t in resp.tools])
            print()


def main() -> None:
    asyncio.run(chat_loop())


if __name__ == "__main__":
    main()