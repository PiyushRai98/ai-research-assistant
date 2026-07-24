"""AI Research Assistant application package.

The package follows Clean Architecture. Dependencies point inwards:

    presentation (backend/, frontend/)
            -> application/   (use cases orchestrating domain logic)
            -> domain/        (entities, value objects, repository ports)
            -> infrastructure/(adapters implementing the ports)
    shared/ holds cross-cutting concerns (config, logging, errors).
"""

__version__ = "0.1.0"
