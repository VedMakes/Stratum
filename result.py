from dataclasses import dataclass, field
from typing import Any


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

    def explain(self, *args, **kwargs):
        if not hasattr(self.model, "explain_model"):
            raise RuntimeError(f"{self.model.__class__.__name__} does not expose explain_model().")
        return self.model.explain_model(*args, **kwargs)

    def plot_optimization(self):
        return self.model.plot_optimization()

    def unwrap(self):
        return self.model
