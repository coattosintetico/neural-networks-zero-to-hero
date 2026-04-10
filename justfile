# Show available tasks
default:
    @just --list

# Sync dependencies into .venv
sync:
    uv sync

# Format all files in place
format:
    uv run ruff check --fix
    uv run ruff format

# Check formatting and linting without modifying files
lint:
    uv run ruff check
    uv run ruff format --check

# Clean build artifacts
clean:
    rm -rf dist/ build/ *.egg-info .ruff_cache .pytest_cache
    find . -type d -name __pycache__ -exec rm -rf {} +

