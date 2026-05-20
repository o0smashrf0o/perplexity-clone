## Project Summary
This project is a single-file, terminal-based “local Perplexity-style” assistant that runs entirely on the user’s machine using Ollama and Agno. The script wires one Agno Agent to multiple tools: SearXNG search, Crawl4AI web crawling, local file access, safe shell commands, and a small Python runner, then exposes a simple CLI chat loop (no HTTP API, no separate frontend). The user types queries into the terminal and gets concise, Markdown-formatted answers with inline citations when web content is used. [github](https://github.com/agno-agi/agno)

## References
https://github.com/agno-agi/agno  
https://docs.agno.com/tools/overview  
https://docs.agno.com/examples/tools/crawl4ai-tools [docs.agno](https://docs.agno.com/examples/tools/crawl4ai-tools)
https://docs.agno.com/tools/attaching-tools [docs.agno](https://docs.agno.com/tools/attaching-tools)
https://docs.agno.com/tools/selecting-tools [docs.agno](https://docs.agno.com/tools/selecting-tools)
https://docs.ollama.com/api/introduction [docs.ollama](https://docs.ollama.com/api/introduction)
https://github.com/unclecode/crawl4ai [github](https://github.com/unclecode/crawl4ai)
https://docs.crawl4ai.com/api/async-webcrawler/ [docs.crawl4ai](https://docs.crawl4ai.com/api/async-webcrawler/)
https://github.com/searxng/searxng  

## Architecture Overview

### Runtime shape
The entire system is a single Python script (for example, `local_perplexity_cli.py`) that the user runs with `python local_perplexity_cli.py`. The script sets up configuration, logging, conversation history, all tools, a unified Agno Agent wired to Ollama, and a simple asynchronous CLI chat loop that reads from `input()` and prints responses. There is no HTTP server, no REST endpoints, and no external state beyond in-memory history; all long-running services (Ollama, SearXNG, Crawl4AI) run separately as local daemons. [docs.crawl4ai](https://docs.crawl4ai.com)

### Config and logging
A small `Config` dataclass at the top of the script centralizes all tunable parameters: Ollama `model_id`, SearXNG base URL and `max_search_results`, Crawl4AI `max_page_length` and `crawl_timeout`, tool limits (`max_tool_calls`, `run_timeout`), and conversational settings like `max_history_turns` and `workspace_dir`. Standard Python logging is configured once and used by all tools to log search queries, crawl operations, and Python execution outcomes for debugging.

### Conversation state and prompt building
Conversation state is held in a `deque` of small dictionaries like `{"user": "...", "assistant": "..."}`, truncated to a fixed number of past turns to control context size. A helper `build_prompt(user_message)` converts this history into a simple textual block (“Turn N – User/Assistant”) followed by the current user message, which becomes the single prompt string passed to `agent.arun()` each turn. This keeps the agent stateless from the framework’s perspective while still giving it short-term context within the script.

### Tools and capabilities
All capabilities are defined as Agno tools inside the same file and passed into a single Agent. The minimal script uses: [docs.agno](https://docs.agno.com/tools/overview)

- `searxng_search`: A `@tool` factory that calls a local SearXNG instance using `httpx` against `/search?format=json`, returning a textual list of top results with title, URL, and snippet; it is the main query → URL discovery mechanism. [docs.crawl4ai](https://docs.crawl4ai.com/core/quickstart/)
- `Crawl4aiTools`: Agno’s built-in wrapper to Crawl4AI for single-URL crawling and Markdown extraction from web pages. [docs.agno](https://docs.agno.com/examples/tools/crawl4ai-tools)
- `batch_web_crawler`: A custom async `@tool` that uses `AsyncWebCrawler` to crawl multiple URLs concurrently, truncating content to `max_page_length` and returning a `dict[url, text]` for efficient multi-page fetch. [github](https://github.com/unclecode/crawl4ai)
- `FileTools`: Agno’s file toolkit, attached as-is but governed by instructions to keep operations within the workspace directory and honor read/write expectations. [docs.agno](https://docs.agno.com/tools/attaching-tools)
- `ShellTools`: Agno’s shell toolkit, used for safe, read-only commands like `ls`, `cat`, and `grep`, with destructive commands prevented via instructions and user intent constraints. [docs.agno](https://docs.agno.com/tools/attaching-tools)
- `python_runner`: A custom `@tool` that runs short Python snippets in a child process (`python -c`), with simple banned patterns (`import os`, `rm -rf`, etc.) and timeouts, intended for numeric calculations and tiny analyses.

Together, these tools give the agent web research, file I/O, shell inspection, and lightweight computation, all accessible through natural language.

### Agent configuration and behavior
A single `build_agent(cfg)` function constructs the `Perplexity-style` agent with `model=Ollama(id=cfg.model_id)`, the tools list above, and a concise `DESCRIPTION` describing the assistant as a local Perplexity-like helper. `build_instructions(cfg)` returns a short list of instructions that encode: when to use web search and Crawl4AI, when to use file and shell tools, when to invoke the Python runner, safety rules (no destructive shell, stay inside workspace), and answer formatting rules (concise Markdown with inline `[Source](URL)` citations when web content is used). A small `sanitize_output()` helper strips out any leaked internal planning markers (like “PHASE” scaffolding) before printing to the user, keeping the output clean. [github](https://github.com/agno-agi/agno)

### CLI chat loop
The main execution path is `chat_loop()`: it builds the agent once, prints a simple banner, and then repeatedly reads user input from the terminal. For each non-empty line that is not “exit” or “quit”, the script builds a prompt from history, calls `await agent.arun(prompt)`, passes the resulting text through `sanitize_output()`, prints the answer, and appends the turn to `history`. If tool use metadata is present, it prints a short “Tools used: [...]” line for transparency, which doubles as lightweight debugging of tool orchestration behavior.

## Build and Test Order

1. **Install runtime and local services**  
   Install Python, `agno`, `httpx`, `crawl4ai`, and set up Ollama with at least one chat-capable model like `llama3:8b`; verify `ollama list` and a simple `/api/generate` call work. Optionally install and configure SearXNG locally and ensure its `/search?format=json` endpoint returns results for basic queries. [docs.agno](https://docs.agno.com)

2. **Create the single-file script with config and history**  
   Create `local_perplexity_cli.py`, add the `Config` dataclass, logging setup, and `history` deque with `build_prompt(user_message)`; run the script and verify prompt-building logic by printing the constructed prompt for a couple of turns.

3. **Add the LLM and basic agent (no tools)**  
   Import Agno and Ollama, implement `build_agent(cfg)` with just `model=Ollama(id=cfg.model_id)` and minimal instructions; temporarily omit tools and implement a trivial `chat_loop()` that calls `agent.arun()` and prints responses to confirm basic LLM connectivity.

4. **Implement and test web tools**  
   Add `make_searxng_tool()` to call local SearXNG, and attach it to the agent; query “Search the web for X using searxng_search and show results” and confirm that the tool is invoked and returns formatted results. Then add `Crawl4aiTools` and `batch_web_crawler`, testing with explicit prompts like “Use batch_web_crawler to fetch these URLs: …” to validate Markdown extraction. [docs.crawl4ai](https://docs.crawl4ai.com/api/async-webcrawler/)

5. **Attach file and shell tools with safety rules**  
   Attach `FileTools()` and `ShellTools()` to the agent and expand `build_instructions()` to emphasize read-only behavior and workspace confinement. Test with prompts like “List files in the current directory” and “Read the contents of file X in the workspace” to confirm both capability and safety. [docs.agno](https://docs.agno.com/tools/overview)

6. **Add and validate the Python runner tool**  
   Implement `python_runner` using a subprocess call with basic banned patterns and a timeout; attach it to the agent and test prompts like “Use python_runner to compute 37**3” to confirm code execution and error reporting work as expected.

7. **Tune instructions, citations, and interaction**  
   Refine `build_instructions()` to encourage using SearXNG + Crawl4AI for web questions and including inline `[Source](URL)` citations based on the tool outputs, and adjust `sanitize_output()` if any internal scaffolding leaks. Run several mixed queries (web research, local file questions, shell environment checks, and Python calculations) and iterate until responses are concise, correctly use tools, and remain safe. [docs.agno](https://docs.agno.com/tools/selecting-tools)
