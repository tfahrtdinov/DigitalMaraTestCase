"""Where parameters and metrics go: the console or an MLflow run. Both are context managers;
`child(name)` gives a nested logger for a sub-run such as one HPO trial."""

import os
from functools import cache
from typing import Protocol

import requests

LOCAL_SERVER = "http://localhost:5001"
LOCAL_STORE = "sqlite:///mlflow.db"


@cache
def tracking_uri() -> str:
    """MLFLOW_TRACKING_URI, by default the server from docker-compose.yml, if that server
    answers; otherwise a local store in the working directory."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", LOCAL_SERVER)
    try:
        requests.get(f"{uri}/health", timeout=2).raise_for_status()
    except requests.RequestException:
        print(f"MLflow server {uri} does not answer, using {LOCAL_STORE}")
        return LOCAL_STORE
    return uri


class Logger(Protocol):
    def log_params(self, params: dict) -> None: ...

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None: ...

    def child(self, name: str) -> "Logger": ...

    def __enter__(self) -> "Logger": ...

    def __exit__(self, *exc) -> None: ...


class ConsoleLogger:
    def __init__(self, name: str = ""):
        self.name = name

    def log_params(self, params: dict) -> None:
        print(f"[{self.name}] params: {params}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        text = ", ".join(f"{k}={v:.4g}" for k, v in metrics.items())
        prefix = f"[{self.name}]" + (f" step {step}:" if step is not None else "")
        print(prefix, text)

    def child(self, name: str) -> "ConsoleLogger":
        return ConsoleLogger(f"{self.name}/{name}")

    def __enter__(self) -> "ConsoleLogger":
        return self

    def __exit__(self, *exc) -> None:
        pass


class MLflowLogger:
    """One MLflow run per `with` block, stored at `tracking_uri()`."""

    def __init__(self, experiment: str, run_name: str, nested: bool = False):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri())
        mlflow.set_experiment(experiment)
        self.experiment, self.run_name, self.nested = experiment, run_name, nested

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def child(self, name: str) -> "MLflowLogger":
        return MLflowLogger(self.experiment, name, nested=True)

    def __enter__(self) -> "MLflowLogger":
        self.mlflow.start_run(run_name=self.run_name, nested=self.nested)
        return self

    def __exit__(self, *exc) -> None:
        self.mlflow.end_run()
