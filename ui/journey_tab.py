"""Deprecated — safe to delete (frontend_only branch).

Already unmounted before this branch (never wired into ui/admin_app.py) and
already broken by the single-strategy collapse (imported pipeline/embedder.py
and pipeline.chunking.REGISTRY, both removed — see CHANGELOG.md). Superseded
entirely by the Next.js port (frontend/).
"""
