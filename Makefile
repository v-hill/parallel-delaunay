.PHONY: test format

# Run the unit tests and the full triangulation check.
test:
	uv run pytest

# Format Python code with Ruff.
format:
	uv run ruff format .
