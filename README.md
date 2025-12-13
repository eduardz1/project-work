# Traveling Goblin Problem

To run the project, use `uv`:

```bash
uv run tgp
```

## CUDA Support

If you have a compatible NVIDIA GPU, to enable CUDA support, add the extra `gpu` dependency to the virtual environment:

```bash
uv sync --extra gpu
```

## Testing

To run the tests, use:

```bash
uv run pytest
```

## Linting

Linting is handled by `ruff` and `ty`. To lint the code, use:

```bash
uv sync --group dev # add '--extra gpu' here for CUDA support
uv run ruff check --fix
uv run ty check
```
