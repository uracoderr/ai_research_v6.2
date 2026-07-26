"""
Shared infrastructure for ThesisPilot: logging, security helpers, session
isolation, report storage, LLM client, and caching.

Nothing in `agents/` should duplicate what lives here - if two agents need
the same plumbing (e.g. calling the LLM, or sanitising a filename), it
belongs in this package.
"""
