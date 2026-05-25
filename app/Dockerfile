# ─────────────────────────────────────────────────
#  Star-Predictor – Production Dockerfile
#  Multi-stage build: keeps the final image small
# ─────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install only what we need to compile wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies into an isolated prefix
COPY app/requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 2: runtime ──────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="star-predictor-team"
LABEL description="GitHub Repository Star-Count Predictor – ML Inference Service"
LABEL version="1.0.0"

WORKDIR /app

# Pull in pre-built packages from builder stage
COPY --from=builder /install /usr/local

# Copy application source and model artefacts
COPY app/predict.py         .
COPY app/ridge_model.pkl    .
COPY app/rf_model.pkl       .
COPY app/gb_model.pkl       .
COPY app/preprocessor.pkl   .

# Non-root user for security hardening
RUN useradd --no-create-home --shell /bin/false appuser
USER appuser

# Smoke-test: verify all imports resolve before the image is finalised
RUN python -c "import pandas, numpy, joblib, sklearn; print('All dependencies OK')"

# Default: run the prediction demo
CMD ["python", "predict.py"]

