---
name: agentcache-research
description: Conducts deep, multi-angle research using the agentcache library's parallel cache-safe forks. Use when the user asks to research a complex topic, investigate architecture trade-offs, or synthesize information from multiple angles.
---

# Agentcache Deep Research

This skill teaches the agent how to leverage the `agentcache` library to perform deep, multi-angle research. Instead of trying to answer complex questions using a single prompt or web search, this skill delegates the work to a swarm of parallel cache-safe worker agents.

## Quick Start

When the user asks you to research a topic, run the provided utility script.

```bash
python examples/agentcache-research-skill/scripts/run_research.py "The user's research topic" --output research_report.md
```

After the script completes, read the generated markdown report and present the findings to the user.

## How it works (for the agent's understanding)

The `run_research.py` script uses `agentcache` to perform the following workflow:

1. **Plan**: A lead `AgentSession` is created with a comprehensive system prompt. It asks the model to break the topic into 3-4 distinct investigation angles.
2. **Parallel Workers**: The script uses `asyncio.gather` to launch parallel `session.fork()` calls. 
   - Each fork uses `ForkPolicy.cache_safe_ephemeral()`
   - This ensures all workers share the exact prompt prefix of the lead session, triggering the provider's prompt cache and saving tokens.
   - The workers do *not* mutate the parent session's memory, ensuring perfect isolation.
3. **Synthesis**: A final cache-safe fork consumes the structured reports from the workers and synthesizes them into an executive brief.

## Output formatting

When presenting the research back to the user:
1. Do not just dump the raw markdown file.
2. Provide a 1-2 paragraph executive summary.
3. Highlight the specific angles that were investigated.
4. Let the user know they can read the full, detailed output in the generated `research_report.md` file.
