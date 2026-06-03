from ..base_model import BaseModel
from skopt.space import Integer, Categorical
import copy
import warnings


class RandomForestModel(BaseModel):
    def __init__(self, mode = None, crossval = False , use_smote=False, verbose=0, supports_cuda=False):
        super().__init__(mode = mode, crossval = crossval, needs_scaling = False, use_smote= use_smote, supports_cuda = supports_cuda, verbose=verbose)
        
        if mode == "Regressor":
            if self._enable_CUDA:
                from cuml.ensemble import RandomForestRegressor as cuRFRegressor
                self._model = cuRFRegressor(random_state=42)
            else:
                from sklearn.ensemble import RandomForestRegressor
                self._model = RandomForestRegressor(random_state=42, n_jobs=self.n_jobs)
        elif mode == "Classifier":
            if self._enable_CUDA:
                from cuml.ensemble import RandomForestClassifier as cuRFClassifier
                self._model = cuRFClassifier(random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier
                self._model = RandomForestClassifier(random_state=42, n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.param_search_space = {
            "n_estimators": Integer(100, 1000),          # Number of trees
            "max_depth": Integer(3, 15),                 # Tree depth
            "min_samples_split": Integer(2, 20),         # Minimum samples to split an internal node
            "min_samples_leaf": Integer(1, 20),          # Minimum samples at a leaf node
            "max_features": Categorical(["sqrt", "log2", None]),  # Number of features to consider for the best split
            "bootstrap": Categorical([True, False]),      # Whether bootstrap samples are used when building trees
            "criterion": Categorical(["gini", "entropy", "log_loss"]) if mode == "Classifier" else Categorical(["squared_error", "absolute_error"]),  # Function to measure the quality of a split
            "verbose": 0, 
            "max_leaf_nodes" : Integer(10, 100),  # Maximum number of leaf nodes
            "class_weight": Categorical([None, "balanced"]) if mode == "Classifier" else None
        }
        
    def explain_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimization first.")
        if self._enable_CUDA:
            raise RuntimeError("RandomForestModel explain_model uses CPU plotting/tree tooling; disable CUDA for explanation.")
        import pandas as pd
        import matplotlib.pyplot as plt
        from dtreeviz.trees import dtreeviz
        
        if not hasattr(self._best_model, "predict") or not callable(self._best_model.predict):
            raise RuntimeError("Best model does not have a predict method.")

        X, y = self.train_X, self.train_y 

        model = copy.deepcopy(self._best_model)
        estimator = self.final_estimator(model)

        importances = estimator.feature_importances_
        feat_names = self.feature_names(X, model)
        
        if isinstance(feat_names, pd.Index):
            feat_names = feat_names.tolist()
            
        feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)

        feat_imp.plot(kind="bar", figsize=(12, 6), title="Feature Importances")
        plt.ylabel("Importance Score")
        plt.tight_layout()
        plt.show()
        
    
        try:
            tree_model = estimator.estimators_[0] if hasattr(estimator, "estimators_") else estimator
            viz = dtreeviz(
                tree_model,
                X,
                y,
                target_name="label" if self._mode == "Classifier" else "value",
                feature_names=self.input_feature_names(X),
                class_names=y.unique().tolist() if self._mode == "Classifier" and hasattr(y, "unique") else None,
            )
            viz.view()
        except Exception as exc:
            warnings.warn(f"Tree visualization skipped: {exc}", UserWarning)

        return feat_imp
