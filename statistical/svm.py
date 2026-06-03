import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.utils import resample
from skopt.space import Categorical, Real

from ..base_model import BaseModel


class SVMModel(BaseModel):
    def __init__(
        self,
        mode=None,
        crossval=False,
        use_smote=False,
        needs_scaling=True,
        proba=False,
        supports_cuda=False,
        verbose=0,
    ):
        super().__init__(
            mode=mode,
            crossval=crossval,
            needs_scaling=needs_scaling,
            use_smote=use_smote,
            supports_cuda=supports_cuda,
            verbose=verbose,
        )
        self._proba = proba

        if mode == "Regressor":
            if proba:
                raise ValueError("Probability estimates are not supported for SVM Regressor.")
            if self._enable_CUDA:
                from cuml.svm import SVR
            else:
                from sklearn.svm import SVR
            self._model = SVR()
        elif mode == "Classifier":
            if self._enable_CUDA and proba:
                raise ValueError("cuML SVC does not expose sklearn-style probability estimates.")
            if self._enable_CUDA:
                from cuml.svm import SVC
                self._model = SVC()
            else:
                from sklearn.svm import SVC
                self._model = SVC(random_state=42, probability=proba)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.param_search_space = {
            "C": Real(1e-3, 1e3, prior="log-uniform"),
            "kernel": Categorical(["linear", "rbf", "poly", "sigmoid"]),
            "gamma": Categorical(["scale", "auto"]),
            "degree": Categorical([2, 3, 4]),
            "coef0": Real(0.0, 5.0),
            "shrinking": Categorical([True, False]),
        }
        if mode == "Classifier":
            self.param_search_space["class_weight"] = Categorical([None, "balanced"])
        if mode == "Regressor":
            self.param_search_space["epsilon"] = Real(1e-3, 1.0, prior="log-uniform")

    def _init_model(self, params, use_cuda=False):
        params = dict(params)
        if params.get("kernel") != "poly":
            params.pop("degree", None)
        if params.get("kernel") not in {"poly", "sigmoid"}:
            params.pop("coef0", None)
        return super()._init_model(params, use_cuda=use_cuda)

    def explain_model(self, sample_fraction=0.2, max_samples=1000, n_repeats=10):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimize or fit first.")
        if self._enable_CUDA:
            raise RuntimeError("SVMModel explain_model uses CPU permutation importance; disable CUDA for explanation.")
        import pandas as pd
        import seaborn as sns
        from matplotlib import pyplot as plt

        X, y = self.train_X, self.train_y
        if X is None or y is None:
            raise RuntimeError("Training data is missing. Call fit or optimize first.")

        X_sample, y_sample = self._sample_for_importance(X, y, sample_fraction, max_samples)
        self._ensure_score_spec()

        result = permutation_importance(
            self._best_model,
            X_sample,
            y_sample,
            scoring=self._score_spec.cv_scorer,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=self.n_jobs,
        )

        importance_df = pd.DataFrame(
            {
                "Feature": self.input_feature_names(X_sample),
                "Importance_Mean": result.importances_mean,
                "Importance_Std": result.importances_std,
            }
        ).sort_values(by="Importance_Mean", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=importance_df,
            x="Importance_Mean",
            y="Feature",
            xerr=importance_df["Importance_Std"],
        )
        plt.title("Permutation Importance")
        plt.xlabel("Decrease in model score")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()

        if self._mode == "Classifier" and hasattr(self._best_model, "decision_function"):
            decision_scores = self._best_model.decision_function(X_sample)
            plt.figure(figsize=(8, 4))
            sns.histplot(np.asarray(decision_scores).ravel(), kde=True)
            plt.title("SVM Decision Scores")
            plt.tight_layout()
            plt.show()

        return importance_df

    def _sample_for_importance(self, X, y, sample_fraction, max_samples):
        n_rows = len(X)
        n_samples = max(1, min(n_rows, max_samples, int(np.ceil(n_rows * sample_fraction))))
        stratify = None
        if self._mode == "Classifier":
            _, counts = np.unique(y, return_counts=True)
            if len(counts) > 1 and np.min(counts) >= 2 and n_samples >= len(counts):
                stratify = y
        return resample(X, y, n_samples=n_samples, random_state=42, stratify=stratify)
