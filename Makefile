.PHONY: setup lint format test

setup:
	uv pip install -r pyproject.toml --extra dev

lint:
	ruff check .

format:
	ruff format .

test:
	pytest
