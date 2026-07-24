# Contributing

Thanks for helping improve the AI Research Assistant.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install ".[ml,ui,dev]"
pre-commit install
```

## Workflow

1. Create a feature branch off `main`.
2. Make focused changes with tests.
3. Run the quality gate locally (below).
4. Open a PR using the template. CI (lint, test, build) must pass.

## Quality gate

```bash
ruff check app tests        # lint
black --check app tests     # formatting
pytest --cov=app            # tests, coverage >= 80%
```

Pre-commit runs ruff + black automatically on staged files.

## Code standards

- Python 3.12+, full type hints, docstrings on public functions/classes.
- Follow SOLID / DRY / KISS. No duplicated logic, no magic numbers, meaningful names.
- Keep business logic in `application`/`domain`; never in Streamlit or routers.
- New infrastructure integrations implement a `domain` port and are wired in
  `app/backend/container.py`.
- Add or update tests for every behavioural change. Prefer the offline stack
  (hashing embeddings, echo LLM) so tests stay fast and deterministic.

## Frontend changes

The UI is generated **exclusively** from [`DESIGN.md`](../DESIGN.md). Do not
introduce colors, fonts, radii, or spacing outside the documented tokens.
Reference component tokens by name (e.g. `color-block-section`, `button-primary`)
and extend `app/frontend/theme.py` / `components.py` rather than hand-rolling CSS
in views.

## Commit messages

Use clear, imperative summaries (e.g. "Add MMR score threshold"). Reference
issues where relevant.
