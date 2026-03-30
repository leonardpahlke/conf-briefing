# List available recipes
default:
    @just --list

# --- Setup ---

# Sync Python dependencies (runs automatically in devshell)
uv-sync:
    uv sync

# --- Pipeline ---

# Collect schedule and recordings for an event
collect event:
    uv run conf-briefing -c events/{{event}}.toml collect

# Generate reports from collected data
report event:
    uv run conf-briefing -c events/{{event}}.toml clean
    uv run conf-briefing -c events/{{event}}.toml analyze
    uv run conf-briefing -c events/{{event}}.toml visualize
    uv run conf-briefing -c events/{{event}}.toml report

# Interactive Q&A about an event
query event question:
    uv run conf-briefing -c events/{{event}}.toml index
    uv run conf-briefing -c events/{{event}}.toml ask "{{question}}"

# --- Docs ---

# Build the mdbook documentation
docs-build:
    cd docs && mdbook build

# Serve the mdbook documentation locally
docs-serve:
    cd docs && mdbook serve --open

# Clean mdbook build output
docs-clean:
    rm -rf docs/book

# --- Dev ---

# Lint and format Python code
lint:
    uv run ruff check conf_briefing/
    uv run ruff format --check conf_briefing/

# Auto-fix lint issues
fix:
    uv run ruff check --fix conf_briefing/
    uv run ruff format conf_briefing/
