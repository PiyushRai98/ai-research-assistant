"""Application layer: use cases orchestrating domain logic via ports.

This layer depends on ``domain`` (entities + ports) and ``shared`` only. It is
wired to concrete infrastructure adapters at composition time (see
``app.backend.dependencies``), never by importing them directly at module
scope.
"""
