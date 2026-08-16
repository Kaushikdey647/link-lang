"""Deprecated — safe to delete.

ProjectEmbeddings (a LangChain Embeddings wrapper around the e5/cohere local
embedding backends) has no remaining caller now that Qdrant Cloud's
server-side inference (Document(text=..., model=...)) is the only embedding
mechanism — see CHANGELOG.md.
"""
