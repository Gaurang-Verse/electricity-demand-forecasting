"""Saving and loading the shipped model artifact.

A model artifact is a directory containing the fitted estimator
(model.joblib) plus a metadata.json recording what it needs to be used
correctly: which feature columns it expects, in what order, the target
column name, the forecast horizon it was trained for, and where its
training data came from. Inference should never hard-code the feature
list separately from what the model was actually trained on -- it loads
this metadata and uses it, so a mismatch fails loudly instead of silently
scoring the wrong columns.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib


@dataclass
class ModelMetadata:
    model_name: str
    feature_columns: list[str]
    target_col: str
    horizon_hours: int
    trained_on: str  # human-readable description of the training window


def save_model(model, metadata: ModelMetadata, out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(asdict(metadata), f, indent=2)


def load_model(model_dir: str | Path):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / "model.joblib")
    with open(model_dir / "metadata.json") as f:
        metadata = ModelMetadata(**json.load(f))
    return model, metadata
