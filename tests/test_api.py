"""Tests for the FastAPI service."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from elec_forecast.api import app
from elec_forecast.artifact import ModelMetadata, save_model
from elec_forecast.features import FEATURE_COLUMNS, HORIZON_HOURS, TARGET_COL, build_features
from elec_forecast.inference import Predictor
from elec_forecast.models import make_model


def _make_hourly(n_hours: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n_hours, freq="h")
    return pd.DataFrame({"datetime": idx, TARGET_COL: rng.uniform(0.1, 5.0, n_hours)})


@pytest.fixture
def hourly() -> pd.DataFrame:
    return _make_hourly()


@pytest.fixture
def model_dir(tmp_path, hourly) -> str:
    features = build_features(hourly)
    model = make_model("lightgbm", {"n_estimators": 20, "random_state": 0})
    model.fit(features[FEATURE_COLUMNS], features[TARGET_COL])

    metadata = ModelMetadata(
        model_name="lightgbm",
        feature_columns=list(FEATURE_COLUMNS),
        target_col=TARGET_COL,
        horizon_hours=HORIZON_HOURS,
        trained_on="api test fixture",
    )
    out_dir = tmp_path / "model"
    save_model(model, metadata, out_dir)
    return str(out_dir)


@pytest.fixture
def fitted_predictor(model_dir) -> Predictor:
    """A Predictor built directly from the fixture model, independent of
    the app -- used as the reference to compare API responses against."""
    return Predictor(model_dir)


@pytest.fixture
def client(monkeypatch, model_dir) -> TestClient:
    """A TestClient that boots the app for real against the fixture model
    directory, via the same MODEL_DIR env var + lifespan startup path the
    app uses in production -- not a dependency override, so /health and
    /forecast are both exercised through their real startup logic."""
    monkeypatch.setenv("MODEL_DIR", model_dir)
    with TestClient(app) as test_client:
        yield test_client


def _history_payload(hourly: pd.DataFrame, cutoff_idx: int = 700):
    origin = hourly["datetime"].iloc[cutoff_idx]
    history = hourly[hourly["datetime"] < origin]
    payload = [
        {"datetime": row.datetime.isoformat(), "target_mean_kw": row.target_mean_kw}
        for row in history.itertuples()
    ]
    return payload, origin


def test_forecast_returns_24_hours_starting_right_after_history(client, hourly):
    payload, origin = _history_payload(hourly)
    response = client.post("/forecast", json={"history": payload})

    assert response.status_code == 200
    body = response.json()
    assert len(body) == HORIZON_HOURS
    assert body[0]["datetime"] == origin.isoformat()


def test_forecast_matches_calling_predictor_directly(client, fitted_predictor, hourly):
    payload, origin = _history_payload(hourly)
    response = client.post("/forecast", json={"history": payload})
    body = response.json()

    history = hourly[hourly["datetime"] < origin]
    expected = fitted_predictor.predict(history, origin=origin)
    actual = [point["predicted_kw"] for point in body]
    assert actual == pytest.approx(expected["predicted_kw"].tolist())


def test_forecast_accepts_an_explicit_origin_matching_the_default(client, hourly):
    payload, origin = _history_payload(hourly)
    response = client.post("/forecast", json={"history": payload, "origin": origin.isoformat()})

    assert response.status_code == 200
    assert response.json()[0]["datetime"] == origin.isoformat()


def test_forecast_rejects_a_gap_in_history(client, hourly):
    payload, _ = _history_payload(hourly)
    del payload[50]

    response = client.post("/forecast", json={"history": payload})

    assert response.status_code == 400
    assert "not contiguous" in response.json()["detail"]


def test_forecast_rejects_history_that_does_not_reach_the_origin(client, hourly):
    """With origin omitted it's derived from the history sent, so it can
    never disagree with a shorter history -- this has to pin origin
    explicitly to the *original* cutoff while sending truncated history,
    to actually create a mismatch between the two."""
    payload, origin = _history_payload(hourly)
    truncated_payload = payload[:-5]

    response = client.post(
        "/forecast", json={"history": truncated_payload, "origin": origin.isoformat()}
    )

    assert response.status_code == 400
    assert "must end exactly at" in response.json()["detail"]


def test_forecast_rejects_empty_history(client):
    response = client.post("/forecast", json={"history": []})
    assert response.status_code == 422


def test_forecast_rejects_a_negative_reading(client, hourly):
    payload, _ = _history_payload(hourly)
    payload[0]["target_mean_kw"] = -1.0

    response = client.post("/forecast", json={"history": payload})

    assert response.status_code == 422


def test_health_reports_ok_when_the_model_loads(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "lightgbm"}


def test_health_reports_unhealthy_when_the_model_directory_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "does-not-exist"))

    with TestClient(app) as unconfigured_client:
        response = unconfigured_client.get("/health")

    assert response.json()["status"] == "unhealthy"


def test_forecast_returns_503_when_the_model_failed_to_load(monkeypatch, tmp_path, hourly):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "does-not-exist"))
    payload, _ = _history_payload(hourly)

    with TestClient(app) as unconfigured_client:
        response = unconfigured_client.post("/forecast", json={"history": payload})

    assert response.status_code == 503


def test_metrics_endpoint_exposes_prometheus_text_format(client, hourly):
    payload, _ = _history_payload(hourly)
    client.post("/forecast", json={"history": payload})

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "forecast_requests_total" in response.text
    assert "forecast_request_latency_seconds" in response.text
