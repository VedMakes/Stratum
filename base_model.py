from abc import ABC
from dataclasses import dataclass
from inspect import Parameter, signature
from typing import Any, Callable, Optional
import copy
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone, is_classifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    make_scorer,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    precision_recall_fscore_support,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ScoreSpec:
    name: str
    metric: Callable
    direction: str
    cv_scorer: Any
    response: Optional[str] = None


class EstimatorScorer:
    """Pickle-friendly scorer for metrics that need probabilities/scores."""

    def __init__(self, metric, direction="maximize", response=None, name="custom_score"):
        self.metric = metric
        self.direction = direction
        self.response = response
        self.name = name

    def __call__(self, estimator, X, y):
        if self.response == "score":
            y_pred = self._predict_score(estimator, X)
        else:
            y_pred = estimator.predict(X)

        value = float(self.metric(self._as_metric_array(y), self._as_metric_array(y_pred)))
        if not np.isfinite(value):
            raise ValueError(f"Scorer '{self.name}' returned a non-finite value: {value}.")
        return -value if self.direction == "minimize" else value

    def _predict_score(self, estimator, X):
        if hasattr(estimator, "predict_proba"):
            try:
                y_score = estimator.predict_proba(X)
            except AttributeError:
                y_score = None
            if y_score is not None:
                if getattr(y_score, "ndim", 1) == 2 and y_score.shape[1] == 2:
                    return y_score[:, 1]
                return y_score
        if hasattr(estimator, "decision_function"):
            return estimator.decision_function(X)
        raise RuntimeError("Selected score needs predict_proba or decision_function.")

    @staticmethod
    def _as_metric_array(values):
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        if hasattr(values, "get"):
            values = values.get()
        return np.asarray(values)


class DummyEncoder(BaseEstimator, TransformerMixin):
    """
    Small one-hot encoder for pandas/cudf frames.

    It intentionally stays simple: fit records the dummy columns, transform aligns
    future data to that same schema.
    """

    def __init__(self, drop_first=True):
        self.drop_first = drop_first
        self.columns_ = None
        self.is_cudf = False
        self.fitted_ = False

    def fit(self, X, y=None):
        self.is_cudf = "cudf" in str(type(X))
        encoded = self._encode(X)
        self.columns_ = encoded.columns if hasattr(encoded, "columns") else None
        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise RuntimeError("DummyEncoder must be fitted before transform.")

        encoded = self._encode(X)
        if self.columns_ is None:
            return encoded
        if not hasattr(encoded, "reindex"):
            return encoded
        if self.is_cudf:
            missing_cols = [col for col in self.columns_ if col not in encoded.columns]
            for col in missing_cols:
                encoded[col] = 0.0
            encoded = encoded[self.columns_]
            for col in encoded.columns:
                encoded[col] = encoded[col].astype("float32")
            return encoded

        return encoded.reindex(columns=self.columns_, fill_value=0).astype("float32")

    def get_feature_names_out(self, input_features=None):
        if self.columns_ is None:
            raise RuntimeError("DummyEncoder must be fitted before feature names are available.")
        return np.asarray(self.columns_, dtype=object)

    def _encode(self, X):
        if not hasattr(X, "select_dtypes"):
            return np.asarray(X, dtype="float32")

        cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        if self.is_cudf:
            import cudf

            encoded = cudf.get_dummies(X, columns=cat_cols, drop_first=self.drop_first)
            for col in encoded.columns:
                encoded[col] = encoded[col].astype("float32")
            return encoded

        import pandas as pd

        return pd.get_dummies(X, drop_first=self.drop_first).astype("float32")


class DataFrameImputer(BaseEstimator, TransformerMixin):
    """Median/mode imputation that preserves pandas columns and dtypes."""

    def __init__(self, categorical_as_category=False):
        self.categorical_as_category = categorical_as_category
        self.fill_values_ = None
        self.numeric_columns_ = None
        self.categorical_columns_ = None
        self.categories_ = None

    def fit(self, X, y=None):
        if not hasattr(X, "columns"):
            self.fill_values_ = None
            return self

        self.numeric_columns_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_columns_ = [col for col in X.columns if col not in self.numeric_columns_]
        self.fill_values_ = {}
        self.categories_ = {}
        for col in self.numeric_columns_:
            clean = X[col].dropna()
            self.fill_values_[col] = float(clean.median()) if len(clean) else 0.0
        for col in self.categorical_columns_:
            clean = X[col].dropna()
            self.fill_values_[col] = clean.mode().iloc[0] if len(clean) else "__missing__"
            if self.categorical_as_category:
                values = clean.astype(str).drop_duplicates().tolist()
                fill = str(self.fill_values_[col])
                self.categories_[col] = values if fill in values else [*values, fill]
        return self

    def transform(self, X):
        if self.fill_values_ is None:
            array = np.asarray(X)
            if not np.issubdtype(array.dtype, np.number):
                return array
            result = array.astype(float, copy=True)
            medians = np.nanmedian(result, axis=0)
            medians = np.where(np.isnan(medians), 0.0, medians)
            rows, cols = np.where(np.isnan(result))
            result[rows, cols] = medians[cols]
            return result

        result = X.copy()
        for col in self.numeric_columns_:
            result[col] = result[col].fillna(self.fill_values_[col])
        for col in self.categorical_columns_:
            fill = self.fill_values_[col]
            if self.categorical_as_category:
                import pandas as pd

                values = result[col].astype("string").fillna(str(fill))
                result[col] = pd.Categorical(values, categories=self.categories_[col])
            else:
                if str(result[col].dtype) == "category" and fill not in result[col].cat.categories:
                    result[col] = result[col].cat.add_categories([fill])
                result[col] = result[col].fillna(fill)
        return result

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        if self.fill_values_ is not None:
            return np.asarray(list(self.fill_values_), dtype=object)
        raise RuntimeError("Input feature names are unavailable.")


class CudaPipeline(BaseEstimator):
    """Small GPU-only pipeline for cuML preprocessing plus a cuML estimator."""

    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler

    def fit(self, X, y=None):
        X_fit = self.scaler.fit_transform(X) if self.scaler is not None else X
        self.model.fit(X_fit, y)
        return self

    def predict(self, X):
        return self.model.predict(self._transform(X))

    def predict_proba(self, X):
        if not hasattr(self.model, "predict_proba"):
            raise AttributeError("The wrapped cuML model does not expose predict_proba.")
        return self.model.predict_proba(self._transform(X))

    def decision_function(self, X):
        if not hasattr(self.model, "decision_function"):
            raise AttributeError("The wrapped cuML model does not expose decision_function.")
        return self.model.decision_function(self._transform(X))

    @property
    def classes_(self):
        return self.model.classes_

    def get_params(self, deep=True):
        params = {"model": self.model, "scaler": self.scaler}
        if deep and hasattr(self.model, "get_params"):
            params.update({f"model__{key}": value for key, value in self.model.get_params().items()})
        return params

    def set_params(self, **params):
        model_params = {}
        for key, value in params.items():
            if key == "model":
                self.model = value
            elif key == "scaler":
                self.scaler = value
            elif key.startswith("model__"):
                model_params[key.split("__", 1)[1]] = value
        if model_params and hasattr(self.model, "set_params"):
            self.model.set_params(**model_params)
        return self

    def _transform(self, X):
        return self.scaler.transform(X) if self.scaler is not None else X


class BaseModel(ABC):
    """
    Shared lifecycle for the model wrappers.

    Subclasses are expected to set:
    - self._model: an unfitted sklearn-like estimator
    - self.param_search_space: dict of skopt.space dimensions or fixed values
    """

    valid_modes = {"Regressor", "Classifier"}

    def __init__(
        self,
        mode,
        crossval=False,
        needs_scaling=False,
        use_smote=False,
        supports_cuda=False,
        verbose=0,
    ):
        self._mode = mode
        self.opt_crossval = crossval
        self.cv = 5
        self.n_jobs = -1
        self.verbose = verbose
        self._standardize = needs_scaling
        self._use_smote = use_smote
        self._smote_enabled = False
        self._requested_CUDA = supports_cuda
        self._enable_CUDA = bool(supports_cuda)

        self.train_X = None
        self.train_y = None
        self.param_search_space = {}
        self.cv_groups = None

        self._model = None
        self._params = {}
        self._score_spec = None
        self._scorer = None
        self._metric_func = None
        self._cv_scorer = None
        self._score_direction = None
        self._score_needs_response = None
        self._optuna_study_params = None
        self._study = None
        self._best_model = None
        self._best_params = None
        self._scaler = StandardScaler() if needs_scaling else None
        self._cuda_feature_maps = {}
        self._cuda_target_map = None
        self._cuda_inverse_target_map = None

        if mode not in self.valid_modes:
            raise ValueError("mode must be either 'Regressor' or 'Classifier'.")
        if supports_cuda:
            self._ensure_cuda_dependencies()

    @property
    def mode(self):
        return self._mode

    @property
    def enable_CUDA(self):
        return self._enable_CUDA

    def set_loss_function(self, function_name_or_callable, direction=None, response=None):
        self._score_spec = self._make_score_spec(function_name_or_callable, direction=direction, response=response)
        self._scorer = self._score_spec.metric
        self._metric_func = self._score_spec.metric
        self._cv_scorer = self._score_spec.cv_scorer
        self._score_direction = self._score_spec.direction
        self._score_needs_response = self._score_spec.response
        return self

    def set_scorer(self, scorer, direction="maximize", response=None):
        return self.set_loss_function(scorer, direction=direction, response=response)

    def set_study_params(self, study_params):
        if not isinstance(study_params, dict):
            raise ValueError("study_params must be a dictionary.")
        if "direction" not in study_params:
            raise ValueError("study_params must include at least a 'direction' key.")
        self._optuna_study_params = study_params
        return self

    def optimize(self, X, y, n_trials=30, timeout=500, fit_best=True):
        self._validate_xy(X, y)
        self._ensure_ready()
        self._ensure_score_spec()

        if self.opt_crossval:
            X_train, y_train = self._prepare_training_data(X, y)
            X_valid = y_valid = None
        elif self._enable_CUDA:
            if self._use_smote:
                raise RuntimeError("SMOTE currently uses a CPU implementation; disable SMOTE for CUDA optimization.")
            X_gpu, y_gpu = self._port_data_to_cuda(X, y, fit=True)
            X_train, y_train, X_valid, y_valid = self._cuda_train_valid_split(X_gpu, y_gpu)
        else:
            X_train_raw, X_valid, y_train_raw, y_valid = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=42,
                stratify=self._stratify_target(y),
            )
            X_train, y_train = self._prepare_training_data(X_train_raw, y_train_raw)

        from optuna import create_study
        from optuna.pruners import MedianPruner
        from optuna.samplers import TPESampler

        def objective(trial):
            params = self._sample_params(trial)

            if self.opt_crossval:
                if self._enable_CUDA:
                    score = self._custom_cross_val_score(self.train_X, self.train_y, params, self.cv)
                else:
                    score = self._cross_val_score(params)
            else:
                model = self._build_model(params, use_cuda=self._enable_CUDA)
                model.fit(X_train, y_train)
                score = self._score_estimator(model, X_valid, y_valid)

            if self.verbose >= 1:
                print(f"[Trial {trial.number}] Score: {score:.5f}")
            if self.verbose >= 2:
                print(f"    Params: {params}")
            return score

        study_kwargs = (
            dict(self._optuna_study_params)
            if self._optuna_study_params
            else {
                "direction": self._score_spec.direction,
                "sampler": TPESampler(seed=42),
                "pruner": MedianPruner(),
            }
        )
        study = create_study(**study_kwargs)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=self.verbose > 0)

        self._study = study
        self._best_params = dict(study.best_params)
        self._best_model = self._build_model(self._best_params, use_cuda=self._enable_CUDA)
        self.train_X, self.train_y = self._prepare_training_data(X, y)

        if fit_best:
            self.fit()

        if self.verbose >= 1:
            print("Best Score:", study.best_value)
            print("Best Params:", study.best_params)

        return self

    def fit(self, X=None, y=None, params=None):
        if X is not None or y is not None:
            self._validate_xy(X, y)
            self._ensure_ready()
            self.train_X, self.train_y = self._prepare_training_data(X, y)

        if self.train_X is None or self.train_y is None:
            raise RuntimeError("Training data is missing. Call optimize(X, y) or fit(X, y).")

        fit_params = params or self._best_params or {}
        self._best_model = self._build_model(fit_params, use_cuda=self._enable_CUDA)
        self._best_model.fit(self.train_X, self.train_y)
        return self

    def fit_best_model(self):
        return self.fit()

    def score(self, X, y):
        if self._best_model is None:
            raise RuntimeError("No model is available. Call optimize or fit first.")
        self._ensure_score_spec()
        if self._enable_CUDA:
            X, y = self._port_data_to_cuda(X, y)
        return self._score_estimator(self._best_model, X, y)

    def predict(self, X):
        if self._best_model is None:
            raise RuntimeError("No model is available. Call optimize or fit first.")
        if self._enable_CUDA:
            X = self._port_X_to_cuda(X)
        predictions = self._best_model.predict(X)
        return self._decode_target_output(predictions) if self._enable_CUDA else predictions

    def predict_proba(self, X):
        if self._best_model is None:
            raise RuntimeError("No model is available. Call optimize or fit first.")
        if not hasattr(self._best_model, "predict_proba"):
            raise RuntimeError("The fitted model does not expose predict_proba.")
        if self._enable_CUDA:
            X = self._port_X_to_cuda(X)
        try:
            return self._best_model.predict_proba(X)
        except AttributeError as exc:
            raise RuntimeError("The fitted model does not expose predict_proba.") from exc

    def get_best_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimize or fit first.")
        return self._best_model

    def get_best_params(self):
        if self._best_params is None:
            raise RuntimeError("No best parameters found. Run optimize first.")
        return self._best_params

    def get_study(self):
        if self._study is None:
            raise RuntimeError("No study available. Run optimize first.")
        return self._study

    def final_estimator(self, model=None):
        model = model or self._best_model
        if model is None:
            raise RuntimeError("No model is available.")
        return model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model

    def transform_features(self, X, model=None):
        model = model or self._best_model
        if model is None:
            raise RuntimeError("No model is available.")
        if hasattr(model, "steps") and len(model.steps) > 1:
            transformed = X
            for _, step in model.steps[:-1]:
                if hasattr(step, "fit_resample"):
                    continue
                if hasattr(step, "transform"):
                    transformed = step.transform(transformed)
            return transformed
        return X

    def feature_names(self, X=None, model=None):
        X = self.train_X if X is None else X
        model = model or self._best_model

        if hasattr(model, "named_steps"):
            encoder = model.named_steps.get("encoder")
            if encoder is not None and getattr(encoder, "columns_", None) is not None:
                return list(encoder.columns_)
            try:
                return list(model[:-1].get_feature_names_out())
            except Exception:
                pass
        if hasattr(X, "columns"):
            return list(X.columns)
        if hasattr(X, "shape"):
            return [f"Feature {i}" for i in range(X.shape[1])]
        return None

    def input_feature_names(self, X=None):
        X = self.train_X if X is None else X
        if hasattr(X, "columns"):
            return list(X.columns)
        if hasattr(X, "shape"):
            return [f"Feature {i}" for i in range(X.shape[1])]
        return None

    def plot_optimization(self):
        study = self.get_study()
        from optuna.visualization import (
            plot_contour,
            plot_optimization_history,
            plot_param_importances,
            plot_slice,
        )

        plot_optimization_history(study).show()
        plot_param_importances(study).show()
        plot_contour(study).show()
        plot_slice(study).show()

    def eval_model(self, X_test, y_test):
        if self._best_model is None:
            raise RuntimeError("No model is available. Call optimize or fit first.")
        if self._enable_CUDA:
            raise RuntimeError("eval_model uses CPU visualization tooling; use evaluate() in CUDA mode.")
        if self._mode == "Classifier":
            return self._eval_classifier(X_test, y_test)
        if self._mode == "Regressor":
            return self._eval_regressor(X_test, y_test)
        raise RuntimeError(f"Unsupported mode '{self._mode}' for evaluation.")

    def evaluate(self, X_test, y_test):
        if self._best_model is None:
            raise RuntimeError("No model is available. Call optimize or fit first.")
        if self._mode == "Classifier":
            return self._classification_metrics(X_test, y_test)
        if self._mode == "Regressor":
            return self._regression_metrics(X_test, y_test)
        raise RuntimeError(f"Unsupported mode '{self._mode}' for evaluation.")

    def _make_score_spec(self, loss, direction=None, response=None):
        if callable(loss):
            self._validate_metric_callable(loss)
            normalized_direction = self._normalize_score_direction(direction or "minimize")
            normalized_response = self._normalize_score_response(response)
            name = getattr(loss, "__name__", "custom_score")
            if normalized_response == "score" and self._mode != "Classifier":
                raise ValueError("response='score' is only valid for classifier probability/decision metrics.")
            cv_scorer = (
                EstimatorScorer(loss, normalized_direction, normalized_response, name)
                if normalized_response == "score"
                else make_scorer(loss, greater_is_better=normalized_direction == "maximize")
            )
            return ScoreSpec(
                name=name,
                metric=loss,
                direction=normalized_direction,
                cv_scorer=cv_scorer,
                response=normalized_response,
            )

        loss_map = {
            "Regressor": {
                "mse": ScoreSpec("mse", mean_squared_error, "minimize", "neg_mean_squared_error"),
                "mae": ScoreSpec("mae", mean_absolute_error, "minimize", "neg_mean_absolute_error"),
                "rmse": ScoreSpec(
                    "rmse",
                    lambda y_true, y_pred: float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "minimize",
                    "neg_root_mean_squared_error",
                ),
                "r2": ScoreSpec("r2", r2_score, "maximize", "r2"),
            },
            "Classifier": {
                "accuracy": ScoreSpec("accuracy", accuracy_score, "maximize", "accuracy"),
                "f1": ScoreSpec(
                    "f1",
                    lambda y_true, y_pred: f1_score(y_true, y_pred, average="weighted"),
                    "maximize",
                    make_scorer(f1_score, average="weighted"),
                ),
                "roc_auc": ScoreSpec(
                    "roc_auc",
                    self._roc_auc_metric,
                    "maximize",
                    "roc_auc",
                    response="score",
                ),
            },
        }

        key = str(loss).strip().lower()
        try:
            spec = copy.copy(loss_map[self._mode][key])
        except KeyError as exc:
            valid = ", ".join(sorted(loss_map[self._mode]))
            raise ValueError(f"Invalid loss function '{loss}' for {self._mode}. Valid options: {valid}.") from exc
        if direction is not None:
            spec.direction = self._normalize_score_direction(direction)
        if response is not None:
            spec.response = self._normalize_score_response(response)
        return spec

    def _roc_auc_metric(self, y_true, y_score):
        if getattr(y_score, "ndim", 1) == 2:
            if y_score.shape[1] == 2:
                y_score = y_score[:, 1]
            else:
                return roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted")
        return roc_auc_score(y_true, y_score)

    def _ensure_score_spec(self):
        if self._score_spec is None:
            self.set_loss_function("accuracy" if self._mode == "Classifier" else "rmse")

    def _validate_metric_callable(self, metric):
        if not callable(metric):
            raise TypeError("Metric/loss must be a callable or a known metric name.")
        try:
            params = signature(metric).parameters
        except (TypeError, ValueError):
            return

        positional = {
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        }
        accepts_varargs = any(param.kind == param.VAR_POSITIONAL for param in params.values())
        positional_count = sum(param.kind in positional for param in params.values())
        required_positional = sum(
            param.kind in positional and param.default is param.empty for param in params.values()
        )
        if not accepts_varargs and positional_count < 2:
            raise TypeError("Custom metric/loss callables must accept at least y_true and y_pred.")
        if required_positional > 2:
            raise TypeError(
                "Custom metric/loss callables should accept y_true and y_pred. "
                "Use keyword defaults or a closure for extra configuration."
            )

    def _normalize_score_direction(self, direction):
        key = str(direction).strip().lower()
        if key in {"max", "maximize", "higher", "higher_is_better"}:
            return "maximize"
        if key in {"min", "minimize", "lower", "lower_is_better"}:
            return "minimize"
        raise ValueError("direction must be 'maximize' or 'minimize'.")

    def _normalize_score_response(self, response):
        if response is None:
            return None
        key = str(response).strip().lower()
        if key in {"predict", "label", "labels", "class"}:
            return None
        if key in {"score", "proba", "probability", "probabilities", "predict_proba", "decision_function"}:
            return "score"
        raise ValueError("response must be 'predict' or 'score'.")

    def _validate_xy(self, X, y):
        if X is None or y is None:
            raise ValueError("X and y must be provided.")
        if not hasattr(X, "shape"):
            raise ValueError("X must be a pandas-like DataFrame or array-like object.")
        if not hasattr(y, "shape"):
            raise ValueError("y must be a pandas-like Series or 1D array.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows.")

    def _ensure_ready(self):
        if self._model is None:
            raise RuntimeError("Subclasses must set self._model before training.")
        if not hasattr(self, "param_search_space"):
            self.param_search_space = {}

    def _prepare_training_data(self, X, y):
        self._smote_enabled = self._check_imbalance_and_prompt(y)
        X_train, y_train = X, y
        if self._enable_CUDA:
            if self._smote_enabled:
                raise RuntimeError("SMOTE currently uses a CPU implementation; disable SMOTE for CUDA training.")
            X_train, y_train = self._port_data_to_cuda(X_train, y_train, fit=True)
            if self.verbose >= 1:
                print("Data ported to GPU using cuDF.")
        self.train_X, self.train_y = X_train, y_train
        return X_train, y_train

    def _sample_params(self, trial):
        from skopt.space import Categorical, Integer, Real

        params = {}
        for name, space in self.param_search_space.items():
            if isinstance(space, Integer):
                params[name] = trial.suggest_int(name, space.low, space.high)
            elif isinstance(space, Real):
                params[name] = trial.suggest_float(
                    name,
                    space.low,
                    space.high,
                    log=getattr(space, "prior", "") in {"log-uniform", "log"},
                )
            elif isinstance(space, Categorical):
                params[name] = trial.suggest_categorical(name, list(space.categories))
            else:
                params[name] = space
        return self._sanitize_params(params)

    def _sanitize_params(self, params):
        valid_params = self._model.get_params() if hasattr(self._model, "get_params") else {}
        if not valid_params:
            return {}
        clean = {key: value for key, value in params.items() if key in valid_params}

        if "random_state" in valid_params and "random_state" not in clean:
            clean["random_state"] = 42
        if not self._enable_CUDA:
            clean.pop("device", None)
            clean.pop("task_type", None)

        return clean

    def _init_model(self, params, use_cuda=False):
        try:
            model = clone(self._model)
        except Exception:
            model = copy.deepcopy(self._model)
        clean_params = self._sanitize_params(params)
        if clean_params and hasattr(model, "set_params"):
            model.set_params(**clean_params)
        return model

    def _build_model(self, params, use_cuda=False):
        model = self._init_model(params, use_cuda=use_cuda)
        if use_cuda:
            return CudaPipeline(model, self._cuda_scaler() if self._standardize else None)

        steps = [(
            "imputer",
            DataFrameImputer(
                categorical_as_category=(
                    self._model_supports_categoricals() and not self._smote_enabled
                )
            ),
        )]
        if not self._model_supports_categoricals() or self._smote_enabled:
            steps.append(("encoder", DummyEncoder()))
        if self._smote_enabled:
            from imblearn.over_sampling import SMOTE

            steps.append(("smote", SMOTE(random_state=42)))
        if self._standardize:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", model))

        if self._smote_enabled:
            from imblearn.pipeline import Pipeline as ImbalancedPipeline

            return ImbalancedPipeline(steps)
        return Pipeline(steps)

    def _cross_val_score(self, params):
        return self._cpu_cross_val_score(self.train_X, self.train_y, params, self.cv)

    def _cpu_cross_val_score(self, X, y, params, cv_splits):
        model = self._build_model(params, use_cuda=False)
        if hasattr(cv_splits, "split"):
            cv_strategy = cv_splits
        elif self._mode == "Classifier":
            cv_strategy = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        else:
            cv_strategy = KFold(n_splits=cv_splits, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv_strategy,
            groups=self.cv_groups,
            scoring=self._score_spec.cv_scorer,
            n_jobs=self.n_jobs,
            error_score="raise",
        )
        score = float(np.mean(scores))
        return -score if self._score_spec.direction == "minimize" else score

    def _score_estimator(self, model, X, y):
        if self._score_spec.response == "score":
            y_pred = None
            if hasattr(model, "predict_proba"):
                try:
                    y_pred = model.predict_proba(X)
                except AttributeError:
                    y_pred = None
            if y_pred is not None:
                if getattr(y_pred, "ndim", 1) == 2 and y_pred.shape[1] == 2:
                    y_pred = y_pred[:, 1]
            elif hasattr(model, "decision_function"):
                try:
                    y_pred = model.decision_function(X)
                except AttributeError as exc:
                    raise RuntimeError("Selected score needs predict_proba or decision_function.") from exc
            else:
                raise RuntimeError("Selected score needs predict_proba or decision_function.")
        else:
            y_pred = model.predict(X)

        if self._enable_CUDA and self._mode == "Classifier" and self._score_spec.response != "score":
            y = self._decode_target_output(y)
            y_pred = self._decode_target_output(y_pred)

        value = float(self._score_spec.metric(self._as_metric_array(y), self._as_metric_array(y_pred)))
        if not np.isfinite(value):
            raise ValueError(f"Scorer '{self._score_spec.name}' returned a non-finite value: {value}.")
        return value

    def _check_imbalance_and_prompt(self, y):
        if self._mode != "Classifier":
            return False
        if not self._enable_CUDA and not is_classifier(self._model):
            return False
        if self._enable_CUDA and self._use_smote is None:
            return False

        _, counts = np.unique(y, return_counts=True)
        if len(counts) < 2:
            return False

        imbalance_ratio = min(counts) / len(y)
        if self._use_smote is not None:
            return bool(self._use_smote)
        if imbalance_ratio >= 0.15:
            return False

        warnings.warn("Detected imbalanced classification data.", UserWarning)
        print(f"Minority class is {imbalance_ratio * 100:.1f}% of samples.")
        answer = input("Apply SMOTE? (yes/no): ").strip().lower()
        self._use_smote = answer in {"yes", "y"}
        return self._use_smote

    def _stratify_target(self, y):
        if self._mode != "Classifier":
            return None
        _, counts = np.unique(y, return_counts=True)
        return y if len(counts) > 1 and np.min(counts) >= 2 else None

    def _classification_metrics(self, X_test, y_test):
        if self._enable_CUDA:
            X_test, y_test = self._port_data_to_cuda(X_test, y_test)
            y_pred = self._best_model.predict(X_test)
        else:
            y_pred = self.predict(X_test)
        y_test_metric = self._as_metric_array(y_test)
        y_pred_metric = self._as_metric_array(y_pred)
        classes = getattr(self._best_model, "classes_", np.unique(y_test_metric))
        classes = self._as_metric_array(classes)
        per_class = precision_recall_fscore_support(
            y_test_metric, y_pred_metric, labels=classes, zero_division=0
        )
        metrics = {
            "accuracy": float(accuracy_score(y_test_metric, y_pred_metric)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test_metric, y_pred_metric)),
            "f1_weighted": float(f1_score(y_test_metric, y_pred_metric, average="weighted", zero_division=0)),
            "precision_weighted": float(
                precision_score(y_test_metric, y_pred_metric, average="weighted", zero_division=0)
            ),
            "recall_weighted": float(recall_score(y_test_metric, y_pred_metric, average="weighted", zero_division=0)),
            "per_class": {
                str(label): {
                    "precision": float(per_class[0][index]),
                    "recall": float(per_class[1][index]),
                    "f1": float(per_class[2][index]),
                    "support": int(per_class[3][index]),
                }
                for index, label in enumerate(classes)
            },
        }

        try:
            if hasattr(self._best_model, "predict_proba"):
                try:
                    y_score = self._best_model.predict_proba(X_test)
                except AttributeError:
                    y_score = None
                if y_score is None:
                    return metrics
                y_score = self._as_metric_array(y_score)
                if getattr(y_score, "ndim", 1) == 2 and y_score.shape[1] == 2:
                    metrics["positive_class"] = str(classes[1])
                    binary_true = (y_test_metric == classes[1]).astype(int)
                    metrics["pr_auc"] = float(average_precision_score(binary_true, y_score[:, 1]))
                    metrics["brier_score"] = float(brier_score_loss(binary_true, y_score[:, 1]))
                    try:
                        metrics["roc_auc"] = float(roc_auc_score(binary_true, y_score[:, 1]))
                    except ValueError:
                        pass
                elif getattr(y_score, "ndim", 1) == 2 and len(classes) > 2:
                    one_hot = np.column_stack([y_test_metric == label for label in classes]).astype(float)
                    metrics["brier_score"] = float(np.mean(np.sum((y_score - one_hot) ** 2, axis=1)))
                    try:
                        metrics["roc_auc_ovr_weighted"] = float(
                            roc_auc_score(y_test_metric, y_score, labels=classes, multi_class="ovr", average="weighted")
                        )
                    except ValueError:
                        pass
        except ValueError:
            pass

        return metrics

    def _regression_metrics(self, X_test, y_test):
        if self._enable_CUDA:
            X_test, y_test = self._port_data_to_cuda(X_test, y_test)
            y_pred = self._best_model.predict(X_test)
        else:
            y_pred = self.predict(X_test)
        y_test_metric = self._as_metric_array(y_test)
        y_pred_metric = self._as_metric_array(y_pred)
        mse = float(mean_squared_error(y_test_metric, y_pred_metric))
        return {
            "mae": float(mean_absolute_error(y_test_metric, y_pred_metric)),
            "mse": mse,
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(y_test_metric, y_pred_metric)),
        }

    def _model_supports_categoricals(self):
        model_name = self._model.__class__.__name__.lower()
        return any(name in model_name for name in ("xgb", "lgbm", "lightgbm"))

    def _port_data_to_cuda(self, X, y, fit=False):
        return self._port_X_to_cuda(X, fit=fit), self._port_y_to_cuda(y, fit=fit)

    def _port_X_to_cuda(self, X, fit=False):
        import cudf

        if not isinstance(X, cudf.DataFrame):
            if hasattr(X, "iloc"):
                X = cudf.DataFrame.from_pandas(X)
            else:
                X = cudf.DataFrame(X)

        for col in X.columns:
            if self._is_categorical_like(X[col].dtype):
                X[col] = self._encode_cuda_series(X[col], self._cuda_feature_maps, col, fit=fit)

        return X.astype("float32")

    def _port_y_to_cuda(self, y, fit=False):
        import cudf

        if not isinstance(y, cudf.Series):
            y = cudf.Series(y)

        if self._mode == "Classifier":
            if fit or self._cuda_target_map is None:
                classes = self._series_values(y)
                self._cuda_target_map = {label: index for index, label in enumerate(classes)}
                self._cuda_inverse_target_map = {index: label for label, index in self._cuda_target_map.items()}
            if y.dtype in ["object", "category"] or self._cuda_target_map is not None:
                y = y.map(self._cuda_target_map)
                if bool(y.isnull().any()):
                    raise ValueError("y contains labels that were not seen during CUDA training.")
            return y.astype("int32")

        return y.astype("float32")

    def _encode_cuda_series(self, series, maps, key, fit=False):
        if fit or key not in maps:
            values = self._series_values(series)
            maps[key] = {value: index for index, value in enumerate(values)}
        return series.map(maps[key]).fillna(-1)

    def _is_categorical_like(self, dtype):
        dtype_name = str(dtype).lower()
        return dtype_name in {"object", "category", "string"} or "str" in dtype_name

    def _series_values(self, series):
        values = series.dropna().unique()
        if hasattr(values, "to_pandas"):
            values = values.to_pandas()
        if hasattr(values, "to_arrow"):
            values = values.to_arrow().to_pylist()
        if hasattr(values, "tolist"):
            values = values.tolist()
        return list(values)

    def _decode_target_output(self, values):
        if self._cuda_inverse_target_map is None:
            return values
        import cudf

        if not isinstance(values, cudf.Series):
            values = cudf.Series(values)
        return values.astype("int32").map(self._cuda_inverse_target_map)

    def _as_metric_array(self, values):
        if hasattr(values, "to_numpy"):
            values = values.to_numpy()
        if hasattr(values, "get"):
            values = values.get()
        return np.asarray(values)

    def _cuda_scaler(self):
        try:
            from cuml.preprocessing import StandardScaler as cuStandardScaler
        except ImportError as exc:
            raise RuntimeError("CUDA scaling requires cuml.preprocessing.StandardScaler.") from exc
        return cuStandardScaler()

    def _cuda_train_valid_split(self, X, y, test_size=0.2):
        import cupy as cp

        n_rows = len(X)
        if n_rows < 2:
            raise ValueError("At least two rows are required for a validation split.")

        indices = cp.random.RandomState(42).permutation(n_rows)
        n_valid = max(1, int(round(n_rows * test_size)))
        n_valid = min(n_valid, n_rows - 1)
        valid_idx = indices[:n_valid]
        train_idx = indices[n_valid:]
        return X.iloc[train_idx], y.iloc[train_idx], X.iloc[valid_idx], y.iloc[valid_idx]

    def _custom_cross_val_score(self, X, y, params, cv_splits):
        import cupy as cp

        n_rows = len(X)
        if cv_splits > n_rows:
            raise ValueError("cv_splits cannot exceed the number of training rows.")

        indices = cp.random.RandomState(42).permutation(n_rows)
        folds = cp.array_split(indices, cv_splits)

        scores = []
        for fold_idx, val_idx in enumerate(folds):
            train_parts = [fold for idx, fold in enumerate(folds) if idx != fold_idx]
            train_idx = cp.concatenate(train_parts)
            model = self._build_model(params, use_cuda=True)
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
            model.fit(X_train, y_train)
            scores.append(self._score_estimator(model, X_val, y_val))

        score = float(np.mean(scores))
        return score

    def _eval_classifier(self, X_test, y_test):
        from yellowbrick.classifier import (
            ClassPredictionError,
            ClassificationReport,
            ConfusionMatrix,
            ROCAUC,
        )

        model = copy.deepcopy(self._best_model)
        class_names = (
            self.train_y.unique().tolist()
            if hasattr(self.train_y, "unique")
            else np.unique(self.train_y).tolist()
        )
        visualizers = [
            ClassificationReport(model, classes=class_names, support=True),
            ConfusionMatrix(model, classes=class_names),
            ClassPredictionError(model, classes=class_names),
        ]

        if hasattr(model, "predict_proba") or hasattr(model, "decision_function"):
            visualizers.append(ROCAUC(model, classes=class_names))

        for viz in visualizers:
            viz.fit(self.train_X, self.train_y)
            viz.score(X_test, y_test)
            viz.show()

    def _eval_regressor(self, X_test, y_test):
        import matplotlib.pyplot as plt
        import statsmodels.api as sm
        from yellowbrick.regressor import PredictionError, ResidualsPlot

        model = copy.deepcopy(self._best_model)

        pred_err = PredictionError(model)
        pred_err.fit(self.train_X, self.train_y)
        pred_err.score(X_test, y_test)
        pred_err.show()

        resid_plot = ResidualsPlot(model)
        resid_plot.fit(self.train_X, self.train_y)
        resid_plot.score(X_test, y_test)
        resid_plot.show()

        residuals = y_test - model.predict(X_test)
        sm.qqplot(residuals, line="45", fit=True)
        plt.title("Q-Q Plot of Residuals")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    @staticmethod
    def _ensure_cuda_dependencies():
        missing = []
        for module_name in ("cupy", "cudf", "cuml"):
            try:
                __import__(module_name)
            except ImportError:
                missing.append(module_name)
        if missing:
            raise RuntimeError(
                "supports_cuda=True was requested, but these CUDA dependencies are missing: "
                f"{', '.join(missing)}. Install RAPIDS/cuML or run with supports_cuda=False."
            )
        return True

    @staticmethod
    def _cuda_available():
        try:
            BaseModel._ensure_cuda_dependencies()

            return True
        except RuntimeError:
            return False
