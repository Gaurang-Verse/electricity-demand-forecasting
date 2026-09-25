# Runs the FastAPI service only -- not the training/backtest pipeline.
# Training happens once, locally, against the real dataset
# (scripts/download_data.py -> preprocess.py -> run_backtest.py ->
# evaluate_holdout.py); this image serves the model that produces, it
# doesn't reproduce the training run. See README "Docker" for why.
FROM python:3.11-slim

# LightGBM's compiled extension needs the OpenMP runtime at import time --
# the same underlying need as Homebrew's libomp on macOS (see README
# "Setup"), just under its Linux name and installed with apt instead.
# Without it, `import lightgbm` fails at container startup.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only pyproject.toml + src/ are needed to install and run the API --
# scripts/, tests/, configs/, docs/, notebooks/ and data/ all stay out of
# the image (see .dockerignore), so a source change here can't accidentally
# ship the raw dataset or the dev-only tooling.
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir .

# The model directory is never baked into the image -- it's a regenerable
# binary artifact (see .gitignore), not a build input. Mount your locally
# trained data/processed/model/ at this path when running the container
# (see README "Docker"); without a mount, the app still starts, but
# /health reports unhealthy and /forecast returns 503, exactly like
# running the API locally with no model present.
ENV MODEL_DIR=/app/data/processed/model

EXPOSE 8000

CMD ["uvicorn", "elec_forecast.api:app", "--host", "0.0.0.0", "--port", "8000"]