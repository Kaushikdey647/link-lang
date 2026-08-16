"""Deprecated — safe to delete (frontend_only branch).

LLM safety check + lexical grounding check — superseded by
frontend/lib/server/guardrails.ts, a direct port (JS's native `\p{L}` Unicode
property escape replaces the `[^\W\d_]` word-matching regex). See CHANGELOG.md.
"""
