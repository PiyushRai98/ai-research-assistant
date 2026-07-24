"""Infrastructure layer: concrete adapters implementing the domain ports.

Modules here own all third-party integrations (SQLite, PyMuPDF, FAISS,
sentence-transformers, HTTP LLM backends). Heavy dependencies are imported
lazily inside functions/constructors so the rest of the codebase can be
imported and unit-tested without them.
"""
