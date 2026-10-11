# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.12 project using a `src` layout. Application code lives in `src/badminton_analysis/`; the current package entry point is `main()` in `__init__.py`. Project metadata and dependencies are declared in `pyproject.toml`, while `uv.lock` pins reproducible versions. Keep experiments, generated datasets, model checkpoints, and local notes out of the package; `scratchpad/` is suitable for disposable local exploration but should not become a dependency of production code. Add tests under `tests/`, mirroring the package structure (for example, `tests/test_cli.py`).

## Build, Test, and Development Commands

- `uv sync`: create or update `.venv` from `pyproject.toml` and `uv.lock`.
- `uv run badminton-analysis`: run the installed command-line entry point.
- `uv run python -m badminton_analysis`: run the package directly.
- `uv build`: produce source and wheel distributions in `dist/`.
- `uv run pytest`: run the test suite after pytest is added as a development dependency.

Use `uv add <package>` for runtime dependencies and `uv add --dev <package>` for development-only tools. Commit the resulting lockfile changes.

## Coding Style & Naming Conventions

Follow PEP 8 with four-space indentation and type annotations on public functions. Use `snake_case` for modules, functions, and variables; `PascalCase` for classes; and `UPPER_SNAKE_CASE` for constants. Prefer small, single-purpose functions, explicit imports, and guard clauses over deeply nested branches. Order imports as standard library, third-party, then local modules. If Ruff is introduced, keep its configuration in `pyproject.toml` and run `uv run ruff check .` before submitting.

## Testing Guidelines

No testing framework or coverage threshold is configured yet. New behavior should include focused pytest tests named `test_<behavior>.py`; test functions should describe outcomes, such as `test_main_prints_greeting`. Keep unit tests deterministic and avoid downloading models or datasets. Mark expensive integration tests clearly and isolate them from the default fast suite.

## Commit & Pull Request Guidelines

The repository has no commit history from which to infer an established convention. Use concise, imperative subjects, preferably Conventional Commit prefixes such as `feat: add rally classifier` or `test: cover empty video input`. Pull requests should explain the motivation, summarize implementation and validation, link relevant issues, and call out dependency or model changes. Include screenshots or sample output for user-visible analysis changes, and do not commit secrets, virtual environments, datasets, or generated model artifacts.

## Progress Logging

After material changes, add a dated entry to `WORK_LOG.md`. Record outcomes, validation, decisions, and follow-ups; omit routine commands.
