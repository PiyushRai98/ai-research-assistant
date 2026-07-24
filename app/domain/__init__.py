"""Domain layer: framework-agnostic entities, value objects, and ports.

Nothing in this package may import from ``infrastructure``, ``application``,
``backend``, or ``frontend``. It depends only on the standard library,
``pydantic`` (for validated value objects), and ``app.shared``.
"""
