FROM debian:trixie
COPY --from=ghcr.io/astral-sh/uv:0.7 /uv /uvx /usr/bin/
RUN useradd -u 1000 -d /app app \
    && apt-get update \
    && apt-get install -y ca-certificates git \
    && rm -rf /var/lib/apt/lists/*


ENV UV_CACHE_DIR=/uv-cache/
ENV UV_PYTHON_INSTALL_DIR=/opt/python/
WORKDIR /app
RUN --mount=type=cache,target=/uv-cache \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project

COPY . /app

RUN --mount=type=cache,target=/uv-cache/ \
	uv sync --locked --compile-bytecode

USER 1000
ENV PATH="/app/.venv/bin:$PATH"

CMD ["python", "llmcord.py"]
