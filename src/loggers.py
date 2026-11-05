"""Where parameters and metrics go: the console or an MLflow run. Both are context managers."""

from typing import Protocol


class Logger(Protocol):
    def log_params(self, params: dict) -> None: ...

    def log_metrics(
        self, metrics: dict[str, float], step: int | None = None
    ) -> None: ...

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

    def __enter__(self) -> "ConsoleLogger":
        return self

    def __exit__(self, *exc) -> None:
        pass


class MLflowLogger:
    """One MLflow run per `with` block; `tracking_uri` None uses the MLflow default."""

    def __init__(self, experiment: str, run_name: str, tracking_uri: str | None = None):
        import mlflow

        self.mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.run_name = run_name

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        self.mlflow.log_metrics(metrics, step=step)

    def __enter__(self) -> "MLflowLogger":
        self.mlflow.start_run(run_name=self.run_name)
        return self

    def __exit__(self, *exc) -> None:
        self.mlflow.end_run()
