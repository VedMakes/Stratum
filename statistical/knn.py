
from ..base_model import BaseModel
from skopt.space import Integer, Categorical
from sklearn.inspection import permutation_importance




class KNNModel(BaseModel):
    def __init__(self, mode = None, crossval = False,use_smote = False, needs_scaling = True, verbose=0, supports_cuda=False):
        super().__init__(mode = mode, crossval = crossval, needs_scaling = needs_scaling, use_smote= use_smote, supports_cuda = supports_cuda, verbose=verbose)
        
        if mode == "Regressor":
            if self._enable_CUDA:
                from cuml.neighbors import KNeighborsRegressor
            else:
                from sklearn.neighbors import KNeighborsRegressor
            self._model = KNeighborsRegressor()
        elif mode == "Classifier":
            if self._enable_CUDA:
                from cuml.neighbors import KNeighborsClassifier
            else:
                from sklearn.neighbors import KNeighborsClassifier
            self._model = KNeighborsClassifier()
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.param_search_space = {
            "n_neighbors": Integer(1, 50),                # Number of neighbors
            "weights": Categorical(["uniform", "distance"]),  # Weighting function
            "leaf_size": Integer(10, 100),               # Leaf size for tree-based algorithms
            "p": Integer(1, 2),                          # Power parameter for Minkowski distance
            "metric": Categorical(["minkowski", "chebyshev", "euclidean", "manhattan"]),  # Distance metric
        }
        
   
    def explain_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimization first.")
        if self._enable_CUDA:
            raise RuntimeError("KNNModel explain_model uses CPU permutation importance; disable CUDA for explanation.")
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        if not hasattr(self._best_model, "predict") or not callable(self._best_model.predict):
            raise RuntimeError("Best model does not have a predict method.")
        
        X, y = self.train_X, self.train_y

        self._ensure_score_spec()
        result = permutation_importance(
            self._best_model,
            X,
            y,
            scoring=self._score_spec.cv_scorer,
            n_repeats=10,
            random_state=42,
            n_jobs=self.n_jobs,
        )
        importance_df = pd.DataFrame({
            "Feature": self.input_feature_names(X),
            "Importance_Mean": result.importances_mean,
            "Importance_Std": result.importances_std,
        }).sort_values("Importance_Mean", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance_df, x="Importance_Mean", y="Feature", xerr=importance_df["Importance_Std"])
        plt.title("Permutation Importance")
        plt.tight_layout()
        plt.show()
        return importance_df
