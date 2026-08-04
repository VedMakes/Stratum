from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TrainingResult:
    model: Any
    task: str
    model_name: str
    tuned: bool = False
    preset: str | None = None
    metric: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def estimator(self):
        return self.model.get_best_model()

    @property
    def best_params(self):
        try:
            return self.model.get_best_params()
        except RuntimeError:
            return {}

    @property
    def study(self):
        try:
            return self.model.get_study()
        except RuntimeError:
            return None

    @property
    def feature_names(self):
        return self.model.feature_names()

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y, metric=None, direction=None, response=None):
        if metric is None:
            return self.model.score(X, y)

        previous = self.model._score_spec
        self.model.set_loss_function(metric, direction=direction, response=response)
        try:
            return self.model.score(X, y)
        finally:
            if previous is not None:
                self.model._score_spec = previous
                self.model._scorer = previous.metric
                self.model._metric_func = previous.metric
                self.model._cv_scorer = previous.cv_scorer
                self.model._score_direction = previous.direction
                self.model._score_needs_response = previous.response
            else:
                self.model._score_spec = None
                self.model._scorer = None
                self.model._metric_func = None
                self.model._cv_scorer = None
                self.model._score_direction = None
                self.model._score_needs_response = None

    def evaluate(self, X, y):
        return self.model.evaluate(X, y)

    def get_feature_importance(self, X=None, y=None, n_repeats=3):
        estimator = self.model.final_estimator(self.estimator)
        names = list(self.feature_names or self.model.input_feature_names(X))
        values = getattr(estimator, "feature_importances_", None)
        if values is None and hasattr(estimator, "coef_"):
            coefficients = np.asarray(estimator.coef_)
            values = (
                np.mean(np.abs(coefficients), axis=0)
                if coefficients.ndim > 1
                else np.abs(coefficients)
            )
        if values is None:
            if X is None or y is None:
                raise ValueError("X and y are required for permutation importance.")
            from sklearn.inspection import permutation_importance

            measured = permutation_importance(
                self.estimator,
                X,
                y,
                n_repeats=n_repeats,
                random_state=42,
            )
            values = measured.importances_mean
            names = list(self.model.input_feature_names(X))
        return sorted(
            [
                {"name": str(name), "importance": float(value)}
                for name, value in zip(names, np.asarray(values).ravel())
            ],
            key=lambda item: abs(item["importance"]),
            reverse=True,
        )

    def summarize_trials(self, limit=5):
        study = self.study
        if study is None:
            return []
        reverse = str(study.direction).lower().endswith("maximize")
        complete = [trial for trial in study.trials if trial.value is not None]
        complete.sort(key=lambda trial: trial.value, reverse=reverse)
        return [
            {"number": trial.number, "value": float(trial.value), "params": dict(trial.params)}
            for trial in complete[:limit]
        ]

    def get_pipeline_spec(self):
        pipeline = self.estimator
        steps = []
        if hasattr(pipeline, "steps"):
            steps = [
                {"name": name, "class": transformer.__class__.__name__}
                for name, transformer in pipeline.steps
            ]
        else:
            steps = [{"name": "model", "class": pipeline.__class__.__name__}]
        return {
            "task": self.task,
            "model_name": self.model_name,
            "feature_names": list(self.model.input_feature_names() or []),
            "transformed_feature_names": list(self.feature_names or []),
            "steps": steps,
        }

    def save(self, path):
        import joblib

        joblib.dump(self.estimator, path)
        return path

    def explain(self, *args, **kwargs):
        if not hasattr(self.model, "explain_model"):
            raise RuntimeError(f"{self.model.__class__.__name__} does not expose explain_model().")
        return self.model.explain_model(*args, **kwargs)

    def plot_optimization(self):
        return self.model.plot_optimization()

    def unwrap(self):
        return self.model
