from skopt.space import Real, Integer, Categorical
from ..base_model import BaseModel
import copy 



class LogRegModel(BaseModel):
    def __init__(self, mode = None, crossval = False,use_smote = False, needs_scaling = True, verbose=0, supports_cuda=False):
        super().__init__(mode = mode, crossval = crossval, needs_scaling = needs_scaling, use_smote= use_smote, supports_cuda = supports_cuda, verbose=verbose)
        
        if mode == "Regressor":
            raise ValueError("LogRegModel is designed for classification tasks only.")
        elif mode == "Classifier":
            if self._enable_CUDA:
                from cuml.linear_model import LogisticRegression
                self._model = LogisticRegression()
            else:
                from sklearn.linear_model import LogisticRegression
                self._model = LogisticRegression(random_state=42, solver='lbfgs', n_jobs=self.n_jobs)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        if self._enable_CUDA:
            self.param_search_space = {
                "C": Real(1e-4, 1e2, prior='log-uniform'),
                "max_iter": Integer(100, 1000),
                "tol": Real(1e-4, 1e-2),
            }
        else:
            self.param_search_space = {
                "C": Real(1e-4, 1e2, prior='log-uniform'),  # Inverse of regularization strength
                "penalty": Categorical(['l2','l1', None]),  # Regularization type
                "solver": Categorical(['lbfgs', 'liblinear', 'saga']),  # Optimization algorithm
                "max_iter": Integer(100, 1000),  # Maximum number of iterations
                "tol": Real(1e-4, 1e-2),  # Tolerance for stopping criteria
                "verbose": Categorical([0, 1])  # Verbosity level
            }


    def _init_model(self, params, use_cuda=False):
        params = dict(params)

        if params.get("penalty") == "l1" and params.get("solver") not in ("liblinear", "saga"):
            params["solver"] = "saga"
        if params.get("penalty") is None and params.get("solver") == "liblinear":
            params["solver"] = "lbfgs"
        if params.get("penalty") is None:
            params.pop("C", None)

        return super()._init_model(params, use_cuda=use_cuda)
        
    
    
    def explain_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimization first.")
        if self._enable_CUDA:
            raise RuntimeError("LogRegModel explain_model renders CPU plots; disable CUDA for explanation.")
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        if not hasattr(self._best_model, "predict") or not callable(self._best_model.predict):
            raise RuntimeError("Best model does not have a predict method.")
        
        X, y = self.train_X, self.train_y
        
        clf = copy.deepcopy(self._best_model)
        
        clf.fit(X, y)

        estimator = self.final_estimator(clf)
        feature_names = self.feature_names(X, clf)

        # Convert to DataFrame
        coef_df = pd.DataFrame(estimator.coef_, columns=feature_names)
        coef_df['Class'] = [f'Class {i}' for i in range(estimator.coef_.shape[0])]
        coef_df = coef_df.set_index('Class')

        # Heatmap
        plt.figure(figsize=(10, 6))
        sns.heatmap(coef_df, annot=True, cmap="coolwarm")
        plt.title("Logistic Regression Coefficients (Multiclass)")
        plt.show()
