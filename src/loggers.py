"""Where parameters and metrics go: the console or an MLflow run. Both are context managers;
`child(name)` gives a nested logger for a sub-run such as one HPO trial."""

import os
from typing import Protocol

from dotenv import load_dotenv


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
    """One MLflow run per `with` block. `tracking_uri` None takes MLFLOW_TRACKING_URI
    from the environment or .env; unset, MLflow writes to ./mlruns."""

    def __init__(
        self,
        experiment: str,
        run_name: str,
        tracking_uri: str | None = None,
        nested: bool = False,
    ):
        import mlflow

        load_dotenv()
        self.mlflow = mlflow
        uri = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)
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
