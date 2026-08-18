# ── Stage 1: build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS fe-build
WORKDIR /app

COPY frontend/package.json frontend/pnpm-lock.yaml ./frontend/
RUN npm install -g pnpm && cd frontend && pnpm install --frozen-lockfile

COPY frontend ./frontend
RUN cd frontend && pnpm build

# ── Stage 2: Python runtime ───────────────────────────────────────────────────
FROM python:3.12-slim
WORKDIR /app

ARG MAXUN_ANALYTICS_BUILD_ARG_SHA=unknown
RUN printf '%s\n' "$MAXUN_ANALYTICS_BUILD_ARG_SHA" > /app/MAXUN_ANALYTICS_IMAGE_BUILD_SHA \
    && chmod 0444 /app/MAXUN_ANALYTICS_IMAGE_BUILD_SHA
ENV ANALYTICS_AGENT_SERVICE_ID=maxun-analytics-agent \
    ANALYTICS_AGENT_RELEASE_REPOSITORY=Fatih0234/maxun-analytics-agent

RUN pip install uv

COPY pyproject.toml uv.lock README.md hatch_build.py ./
COPY backend/ backend/

RUN uv sync --no-dev
# Add venv and uv tool bin dirs to PATH.
# uv tool install puts binaries in ~/.local/bin (i.e. /root/.local/bin for root).
ENV PATH="/app/.venv/bin:/root/.local/bin:$PATH"

# Pre-install connector packages so they're available offline at runtime.
# Override at build time: docker build --build-arg CONNECTORS="snowflake bigquery"
# Set to empty string to build a connector-free image: --build-arg CONNECTORS=""
ARG CONNECTORS="snowflake bigquery"
COPY connectors/ connectors/
RUN if [ -n "$CONNECTORS" ]; then \
      for c in $CONNECTORS; do \
        if [ -d "connectors/$c" ]; then \
          echo "→ Installing connector from local source: $c" && \
          uv tool install "connectors/$c"; \
        else \
          echo "→ Installing connector from PyPI: analytics-agent-connector-$c" && \
          uv tool install "analytics-agent-connector-$c"; \
        fi; \
      done; \
    fi

COPY alembic.ini config.yaml.example ./
# Use example as the default config; operators override via volume mount or ENGINES_CONFIG env var
RUN cp config.yaml.example config.yaml

# Copy built SPA — FastAPI serves it via static files + SPA fallback
COPY --from=fe-build /app/frontend/dist ./frontend/dist

# The Maxun deployment runs the image without root. Named volumes are
# initialized from these service-owned directories and remain writable by the
# runtime UID; the root filesystem itself is made read-only by deployment.
RUN groupadd --system --gid 10001 analytics-agent \
    && useradd --system --uid 10001 --gid 10001 --create-home --home-dir /home/analytics-agent analytics-agent \
    && mkdir -p /var/lib/maxun-analytics/materializations /var/lib/maxun-analytics/config \
    && chown -R analytics-agent:analytics-agent /var/lib/maxun-analytics /home/analytics-agent
ENV HOME=/home/analytics-agent \
    PATH="/app/.venv/bin:/home/analytics-agent/.local/bin:/root/.local/bin:$PATH"
USER analytics-agent

EXPOSE 8100

CMD ["sh", "-c", "analytics-agent bootstrap && uvicorn analytics_agent.main:app --host 0.0.0.0 --port 8100"]
