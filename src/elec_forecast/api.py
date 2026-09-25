"""FastAPI service wrapping Predictor.

POST /forecast takes a window of recent hourly history and returns the
next HORIZON_HOURS predictions. GET /health reports whether the model
loaded successfully at startup -- for readiness/liveness probes. GET
/metrics exposes Prometheus counters and a latency histogram for request
volume -- the minimum needed to notice "the API is being hit and behaving
oddly" before Phase 16's forecast-accuracy monitoring exists; this does
not measure forecast accuracy, only that the service is being called and
responding.

The model directory is read once at startup from the MODEL_DIR
environment variable (default: data/processed/model), so the same image
can be pointed at a different model artifact without a rebuild. If
loading fails, the app still starts (so orchestrators can see it and read
/health) but /forecast returns 503 until a working model is available.
"""

import os
import time
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from elec_forecast.features import TARGET_COL
from elec_forecast.inference import Predictor

DEFAULT_MODEL_DIR = "data/processed/model"

REQUEST_COUNT = Counter(
    "forecast_requests_total", "Total /forecast requests, by outcome", ["outcome"]
)
REQUEST_LATENCY = Histogram(
    "forecast_request_latency_seconds", "Time spent handling /forecast requests"
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    model_dir = os.environ.get("MODEL_DIR", DEFAULT_MODEL_DIR)
    try:
        app.state.predictor = Predictor(model_dir)
        app.state.predictor_error = None
    except Exception as exc:
        # Any load failure (missing directory, metadata mismatch, corrupt
        # artifact) should degrade to an unhealthy /health response, not
        # crash the process -- an orchestrator needs a live endpoint to
        # even know something is wrong.
        app.state.predictor = None
        app.state.predictor_error = str(exc)
    yield


app = FastAPI(
    title="Household Electricity Demand Forecasting",
    description="24-hour-ahead hourly power forecast from a window of recent history.",
    version="0.1.0",
    lifespan=lifespan,
)


class HistoryPoint(BaseModel):
    datetime: datetime
    target_mean_kw: float = Field(ge=0, description="Observed hourly mean active power, kW")


class ForecastRequest(BaseModel):
    history: list[HistoryPoint] = Field(
        min_length=1,
        description="Observed hourly readings, ending at origin - 1h. Order doesn't "
        "matter; they're sorted internally.",
    )
    origin: datetime | None = Field(
        default=None,
        description="First hour to forecast. Defaults to one hour after the last "
        "history point.",
    )


class ForecastPoint(BaseModel):
    datetime: datetime
    predicted_kw: float


def get_predictor(request: Request) -> Predictor:
    predictor = request.app.state.predictor
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded: {request.app.state.predictor_error}",
        )
    return predictor


def _history_to_frame(history: Iterable[HistoryPoint]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "datetime": [pd.Timestamp(p.datetime) for p in history],
            TARGET_COL: [p.target_mean_kw for p in history],
        }
    )


@app.post("/forecast", response_model=list[ForecastPoint])
def forecast(
    payload: ForecastRequest, predictor: Predictor = Depends(get_predictor)
) -> list[ForecastPoint]:
    history = _history_to_frame(payload.history)
    origin = pd.Timestamp(payload.origin) if payload.origin is not None else None

    start = time.perf_counter()
    try:
        result = predictor.predict(history, origin=origin)
    except ValueError as exc:
        REQUEST_COUNT.labels(outcome="rejected").inc()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        REQUEST_LATENCY.observe(time.perf_counter() - start)

    REQUEST_COUNT.labels(outcome="success").inc()
    return [
        ForecastPoint(datetime=row.datetime, predicted_kw=float(row.predicted_kw))
        for row in result.itertuples()
    ]


@app.get("/health")
def health(request: Request) -> dict:
    if request.app.state.predictor is None:
        return {"status": "unhealthy", "detail": request.app.state.predictor_error}
    return {"status": "ok", "model": request.app.state.predictor.metadata.model_name}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
