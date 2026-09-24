"""MLflow logging for backtest runs.

One MLflow run per model. Per-fold metrics are logged with the fold index
as the step, so the MLflow UI plots error across the year of folds.

Tracking goes to a local SQLite file (mlflow.db, gitignored). MLflow 3.x
no longer supports the old plain-folder ./mlruns store. View runs with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import mlflow

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "elec-forecast-backtest"


def log_backtest(
    model_name: str,
    model_params: dict,
    summary: dict,
    backtest_config: dict,
    tracking_uri: str = TRACKING_URI,
    experiment: str = EXPERIMENT,
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=model_name):
        mlflow.set_tags({
            "model": model_name,
            "evaluation": "walk-forward backtest on dev; holdout not used",
        })
        mlflow.log_params({f"model.{k}": v for k, v in model_params.items()})
        mlflow.log_params({f"backtest.{k}": v for k, v in backtest_config.items()})

        overall = summary["overall"]
        mlflow.log_metrics({f"overall_{k}": v for k, v in overall.items() if k != "n"})
        for season, m in summary["by_season"].items():
            mlflow.log_metrics({f"{season}_rmse": m["rmse"], f"{season}_mape": m["mape"]})
        for f in summary["per_fold"]:
            mlflow.log_metric("fold_rmse", f["rmse"], step=f["fold"])
            mlflow.log_metric("fold_mape", f["mape"], step=f["fold"])
