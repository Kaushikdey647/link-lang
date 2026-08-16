"""Deprecated — safe to delete.

This one-time migration (add a BM25 sparse vector space to a pre-RRF
english_query collection, recomputing sparse vectors locally via fastembed)
is obsolete: every collection created by the current indexing code already
includes the sparse vector space from its first upsert (Qdrant Cloud
computes it server-side — see pipeline/indexer.py), and local fastembed BM25
computation was removed along with the other local/API embedding backends.
See CHANGELOG.md.
"""
