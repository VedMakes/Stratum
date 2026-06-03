from dataclasses import dataclass
from itertools import combinations
from inspect import signature

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


@dataclass(frozen=True)
class EnsembleCandidate:
    subset: tuple[int, ...]
    strategy: str
    score: float


class Ensemble:
    """
    Select the best ensemble from a set of fitted classification models.

    Pass fitted models plus validation data, then call:
    - ensemble_auto("auto") to search subsets and all usable voting strategies
    - ensemble_auto("soft") to search subsets with one voting strategy
    - ensemble_manual("hard") to use all models with one voting strategy
    """

    strategy_aliases = {
        "auto": "auto",
        "hard": "hard",
        "majority": "hard",
        "soft": "soft",
        "average": "soft",
        "avg": "soft",
        "stack": "stacking",
        "stacking": "stacking",
    }

    def __init__(
        self,
        models: list,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        scorer=accuracy_score,
        loss_function=None,
        score_direction=None,
        scorer_response="predict",
        meta_learner=None,
    ):
        self.models = list(models)
        self.train = X_train
        self.test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.scorer = loss_function or scorer
        self.scorer_name = getattr(self.scorer, "__name__", "custom_scorer")
        self.score_direction = self._normalize_direction(
            score_direction or ("minimize" if loss_function is not None else "maximize")
        )
        self.scorer_response = self._normalize_response(scorer_response)
        self._scorer_accepts_proba = self._callable_accepts("y_proba")
        self.meta_learner = meta_learner or LogisticRegression(max_iter=1000)

        self.strategies = ["hard", "soft", "stacking"]
        self.best_subset = None
        self.best_strategy = None
        self.best_accuracy = 0
        self.best_score = 0
        self.candidates_ = []
        self._fitted_meta_learner = None
        self._classes_ = None
        self._validate_scorer()

    @property
    def selected_models(self):
        if self.best_subset is None:
            return []
        return [self.models[i] for i in self.best_subset]

    @property
    def best_models(self):
        return self.selected_models

    def fit(self, X_val=None, y_val=None, voting_type="auto", max_models=None):
        return self.select(voting_type=voting_type, X_val=X_val, y_val=y_val, max_models=max_models)

    def set_loss_function(self, loss_function, direction="minimize", response="predict"):
        self.scorer = loss_function
        self.scorer_name = getattr(self.scorer, "__name__", "custom_loss")
        self.score_direction = self._normalize_direction(direction)
        self.scorer_response = self._normalize_response(response)
        self._scorer_accepts_proba = self._callable_accepts("y_proba")
        self._validate_scorer()
        self.best_subset = None
        self.best_strategy = None
        self.best_accuracy = 0
        self.best_score = 0
        self.candidates_ = []
        self._fitted_meta_learner = None
        return self

    def set_scorer(self, scorer, direction="maximize", response="predict"):
        return self.set_loss_function(scorer, direction=direction, response=response)

    def ensemble_manual(self, voting_type: str):
        strategy = self._normalize_strategy(voting_type)
        if strategy == "auto":
            raise ValueError("ensemble_manual requires a concrete strategy: hard, soft, or stacking.")
        if strategy == "hard" and self.scorer_response == "proba":
            raise ValueError("Hard voting cannot be used with scorer_response='proba'.")
        self._validate_models()
        self._validate_strategy_for_models(strategy, self.models)

        selected = tuple(range(len(self.models)))
        score = (
            self._evaluate_candidate(selected, strategy, self.test, self.y_test)
            if self.test is not None and self.y_test is not None
            else np.nan
        )
        self._set_best(EnsembleCandidate(selected, strategy, score))
        return self

    def ensemble_auto(self, voting_type="auto", max_models=None):
        return self.select(voting_type=voting_type, max_models=max_models)

    def select(self, voting_type="auto", X_val=None, y_val=None, max_models=None):
        self._validate_models()
        X_val = self.test if X_val is None else X_val
        y_val = self.y_test if y_val is None else y_val
        if X_val is None or y_val is None:
            raise ValueError("Validation data is required. Pass X_test/y_test to __init__ or X_val/y_val to select().")

        self._classes_ = None
        self._fitted_meta_learner = None
        strategies = self._strategies_for(voting_type)
        max_models = len(self.models) if max_models is None else min(max_models, len(self.models))
        if max_models < 1:
            raise ValueError("max_models must be at least 1.")

        candidates = []
        for subset_size in range(1, max_models + 1):
            for subset in combinations(range(len(self.models)), subset_size):
                for strategy in strategies:
                    score = self._evaluate_candidate(subset, strategy, X_val, y_val)
                    if np.isfinite(score):
                        candidates.append(EnsembleCandidate(subset, strategy, score))

        if not candidates:
            raise RuntimeError("No valid ensemble candidate could be evaluated.")

        candidates.sort(key=lambda candidate: candidate.score, reverse=self.score_direction == "maximize")
        self.candidates_ = candidates
        self._set_best(candidates[0])
        return self

    def predict(self, X):
        self._ensure_selected()
        models = self.selected_models

        if self.best_strategy == "stacking":
            meta = self._fitted_meta_learner or self._fit_stacker(models, store=True)
            return meta.predict(self._stacked_probabilities(models, X))
        if self.best_strategy == "soft":
            return self._soft_vote(models, X)
        return self._hard_vote(models, X)

    def predict_proba(self, X):
        self._ensure_selected()
        models = self.selected_models

        if self.best_strategy == "stacking":
            meta = self._fitted_meta_learner or self._fit_stacker(models, store=True)
            if not hasattr(meta, "predict_proba"):
                raise RuntimeError("The fitted meta learner does not expose predict_proba.")
            return meta.predict_proba(self._stacked_probabilities(models, X))
        if not self._all_support_predict_proba(models):
            raise RuntimeError("predict_proba requires soft voting or models that expose predict_proba.")
        return self._average_probabilities(models, X)

    def score(self, X=None, y=None):
        self._ensure_selected()
        X = self.test if X is None else X
        y = self.y_test if y is None else y
        if X is None or y is None:
            raise ValueError("X and y are required for scoring.")
        return self._score_predictions(y, self.predict(X), self._predict_proba_or_none(X))

    def summary(self, top_n=5):
        self._ensure_selected()
        return {
            "best_subset": self.best_subset,
            "best_strategy": self.best_strategy,
            "best_score": self.best_score,
            "score_direction": self.score_direction,
            "scorer": self.scorer_name,
            "selected_models": [self._model_name(self.models[i]) for i in self.best_subset],
            "top_candidates": [
                {
                    "subset": candidate.subset,
                    "strategy": candidate.strategy,
                    "score": candidate.score,
                    "models": [self._model_name(self.models[i]) for i in candidate.subset],
                }
                for candidate in self.candidates_[:top_n]
            ],
        }

    def _evaluate_candidate(self, subset, strategy, X, y):
        models = [self.models[i] for i in subset]

        if strategy == "stacking":
            if self._contains_cuda_model(models):
                return -np.inf
            if self.train is None or self.y_train is None or not self._all_support_predict_proba(models):
                return -np.inf
            try:
                meta = self._fit_stacker(models, store=False)
                stack_features = self._stacked_probabilities(models, X)
            except (AttributeError, RuntimeError):
                return -np.inf
            y_pred = meta.predict(stack_features)
            y_proba = meta.predict_proba(stack_features) if hasattr(meta, "predict_proba") else None
            return self._score_predictions(y, y_pred, y_proba)

        if strategy == "soft":
            if not self._all_support_predict_proba(models):
                return -np.inf
            try:
                y_proba = self._average_probabilities(models, X)
                y_pred = self._classes(models)[np.argmax(y_proba, axis=1)]
            except (AttributeError, RuntimeError):
                return -np.inf
            return self._score_predictions(y, y_pred, y_proba)

        if strategy == "hard":
            if self.scorer_response == "proba":
                return -np.inf
            y_pred = self._hard_vote(models, X)
            return self._score_predictions(y, y_pred)

        raise ValueError(f"Unsupported ensemble strategy '{strategy}'.")

    def _set_best(self, candidate):
        self.best_subset = tuple(candidate.subset)
        self.best_strategy = candidate.strategy
        self.best_accuracy = candidate.score
        self.best_score = candidate.score
        self._classes_ = None
        self._classes_ = self._classes(self.selected_models)
        self._fitted_meta_learner = None
        if self.best_strategy == "stacking":
            self._fit_stacker(self.selected_models, store=True)
        if not self.candidates_:
            self.candidates_ = [candidate]

    def _fit_stacker(self, models, store=False):
        stack_train = self._stacked_probabilities(models, self.train)
        meta = clone(self.meta_learner)
        meta.fit(stack_train, self.y_train)
        if store:
            self._fitted_meta_learner = meta
        return meta

    def _hard_vote(self, models, X):
        predictions = np.column_stack([self._as_numpy(self._predict_model(model, X)) for model in models])
        voted = []
        for row in predictions:
            labels, counts = np.unique(row, return_counts=True)
            voted.append(labels[np.argmax(counts)])
        return np.asarray(voted)

    def _soft_vote(self, models, X):
        probabilities = self._average_probabilities(models, X)
        return self._classes(models)[np.argmax(probabilities, axis=1)]

    def _average_probabilities(self, models, X):
        classes = self._classes(models)
        aligned = [self._aligned_probabilities(model, X, classes) for model in models]
        return np.mean(aligned, axis=0)

    def _stacked_probabilities(self, models, X):
        classes = self._classes(models)
        return np.column_stack([self._aligned_probabilities(model, X, classes) for model in models])

    def _aligned_probabilities(self, model, X, classes):
        model_classes = self._model_classes(model, classes)
        probs = self._as_numpy(self._predict_proba_model(model, X))
        frame = np.zeros((len(probs), len(classes)))
        for src_idx, label in enumerate(model_classes):
            matches = np.where(classes == label)[0]
            if len(matches) == 0:
                raise RuntimeError(f"Model predicted an unknown class label: {label!r}.")
            frame[:, matches[0]] = probs[:, src_idx]
        return frame

    def _all_support_predict_proba(self, models):
        return all(self._supports_predict_proba(model) for model in models)

    def _classes(self, models):
        if self._classes_ is not None:
            return self._classes_

        labels = []
        for model in models:
            model_classes = self._model_classes(model, None)
            if model_classes is not None:
                labels.extend(list(self._as_numpy(model_classes)))
        if labels:
            classes = []
            for label in labels:
                if not any(label == known for known in classes):
                    classes.append(label)
            return np.asarray(classes)
        if self.y_train is not None:
            return np.unique(self.y_train)
        if self.y_test is not None:
            return np.unique(self.y_test)
        raise RuntimeError("Could not infer class labels from models or targets.")

    def _validate_models(self):
        if not self.models:
            raise ValueError("At least one fitted model is required for ensembling.")
        for index, model in enumerate(self.models):
            if not self._supports_predict(model):
                raise ValueError(
                    f"Model at index {index} does not expose predict. "
                    "Fit the model before passing it to Ensemble."
                )

    def _validate_strategy_for_models(self, strategy, models):
        if strategy == "stacking" and self._contains_cuda_model(models):
            raise RuntimeError(
                "Stacking currently fits a CPU meta learner. Use hard/soft voting for CUDA-backed models, "
                "or run the ensemble with CPU models."
            )

    def _contains_cuda_model(self, models):
        return any(self._model_uses_cuda(model) for model in models)

    def _model_uses_cuda(self, model):
        if bool(getattr(model, "_enable_CUDA", False)):
            return True
        wrapped = getattr(model, "model", None)
        if wrapped is not None and bool(getattr(wrapped, "_enable_CUDA", False)):
            return True
        unwrapped = self._unwrap_model(model)
        return bool(getattr(unwrapped, "_enable_CUDA", False))

    def _validate_scorer(self):
        if not callable(self.scorer):
            raise TypeError("scorer/loss_function must be callable.")
        if self.scorer_response == "proba" and not self._scorer_accepts_proba:
            raise ValueError(
                "scorer_response='proba' requires a callable that accepts y_proba, "
                "for example loss(y_true, y_pred, y_proba=None)."
            )
        if self.y_test is not None:
            y_probe = self._as_numpy(self.y_test)
            y_pred_probe = y_probe.copy()
            y_proba_probe = None
            if self.scorer_response == "proba":
                classes = np.unique(y_probe)
                y_proba_probe = np.zeros((len(y_probe), max(1, len(classes))))
            self._score_predictions(y_probe, y_pred_probe, y_proba_probe)

    def _unwrap_model(self, model):
        if hasattr(model, "estimator"):
            try:
                return model.estimator
            except RuntimeError:
                pass
        if hasattr(model, "get_best_model"):
            try:
                return model.get_best_model()
            except RuntimeError:
                pass
        fitted_model = getattr(model, "_best_model", None)
        return fitted_model if fitted_model is not None else model

    def _predict_model(self, model, X):
        if hasattr(model, "predict"):
            return model.predict(X)
        return self._unwrap_model(model).predict(X)

    def _predict_proba_model(self, model, X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return self._unwrap_model(model).predict_proba(X)

    def _supports_predict(self, model):
        return hasattr(model, "predict") or hasattr(self._unwrap_model(model), "predict")

    def _supports_predict_proba(self, model):
        return hasattr(model, "predict_proba") or hasattr(self._unwrap_model(model), "predict_proba")

    def _model_classes(self, model, fallback):
        if hasattr(model, "_cuda_inverse_target_map") and model._cuda_inverse_target_map:
            inverse = model._cuda_inverse_target_map
            return np.asarray([inverse[index] for index in sorted(inverse)])
        if hasattr(model, "model") and hasattr(model.model, "_cuda_inverse_target_map") and model.model._cuda_inverse_target_map:
            inverse = model.model._cuda_inverse_target_map
            return np.asarray([inverse[index] for index in sorted(inverse)])
        fitted = self._unwrap_model(model)
        if hasattr(fitted, "classes_"):
            return fitted.classes_
        return fallback

    def _predict_proba_or_none(self, X):
        try:
            return self.predict_proba(X)
        except (AttributeError, RuntimeError):
            return None

    def _score_predictions(self, y_true, y_pred, y_proba=None):
        y_true = self._as_numpy(y_true)
        y_pred = self._as_numpy(y_pred)
        if y_proba is not None:
            y_proba = self._as_numpy(y_proba)

        if self.scorer_response == "proba" or (self._scorer_accepts_proba and y_proba is not None):
            value = self._call_scorer_with_proba(y_true, y_pred, y_proba)
        else:
            value = self.scorer(y_true, y_pred)

        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"Scorer '{self.scorer_name}' returned a non-finite value: {value}.")
        return value

    def _as_numpy(self, values):
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        if hasattr(values, "get"):
            values = values.get()
        return np.asarray(values)

    def _callable_accepts(self, parameter_name):
        try:
            params = signature(self.scorer).parameters
        except (TypeError, ValueError):
            return False
        return (
            parameter_name in params
            or any(param.kind == param.VAR_KEYWORD for param in params.values())
        )

    def _call_scorer_with_proba(self, y_true, y_pred, y_proba):
        try:
            return self.scorer(y_true, y_pred, y_proba=y_proba)
        except TypeError:
            return self.scorer(y_true, y_pred, y_proba)

    def _normalize_direction(self, direction):
        key = str(direction).strip().lower()
        if key in {"max", "maximize", "higher", "higher_is_better"}:
            return "maximize"
        if key in {"min", "minimize", "lower", "lower_is_better"}:
            return "minimize"
        raise ValueError("score_direction must be 'maximize' or 'minimize'.")

    def _normalize_response(self, response):
        key = str(response).strip().lower()
        if key in {"predict", "label", "labels", "class"}:
            return "predict"
        if key in {"proba", "probability", "probabilities", "predict_proba"}:
            return "proba"
        raise ValueError("scorer_response must be 'predict' or 'proba'.")

    def _strategies_for(self, voting_type):
        strategy = self._normalize_strategy(voting_type)
        if strategy == "auto":
            return list(self.strategies)
        return [strategy]

    def _normalize_strategy(self, voting_type):
        key = str(voting_type).strip().lower().replace("-", "_").replace(" ", "_")
        try:
            return self.strategy_aliases[key]
        except KeyError as exc:
            valid = ", ".join(sorted(self.strategy_aliases))
            raise ValueError(f"Unknown voting_type '{voting_type}'. Valid options: {valid}.") from exc

    def _ensure_selected(self):
        if self.best_subset is None or self.best_strategy is None:
            raise RuntimeError("Run ensemble_auto(), ensemble_manual(), or select() before prediction.")

    def _model_name(self, model):
        if hasattr(model, "model_name"):
            return model.model_name
        return model.__class__.__name__
