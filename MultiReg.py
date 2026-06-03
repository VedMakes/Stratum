from sklearn.linear_model import ElasticNet
from skopt.space import Integer, Real, Categorical
from .base_model import BaseModel
import copy
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


class MultiRegModel(BaseModel):
    def __init__(self, mode= None, crossval= False,use_smote = False, needs_scaling = True , verbose=0, supports_cuda=False):
        if supports_cuda:
            raise ValueError("MultiRegModel does not yet support an all-GPU cuML pipeline.")
        super().__init__(mode = mode, crossval = crossval, needs_scaling = needs_scaling, use_smote= use_smote, supports_cuda = False, verbose=verbose)
        
        degree = 2  # Default degree for polynomial features
        
        if mode == "Regressor":
            self._model = Pipeline([
                ("poly", PolynomialFeatures(degree=degree)),
                ("regressor", ElasticNet(random_state=42, max_iter=10000))
            ])
            
        elif mode == "Classifier":
            raise ValueError("MultiRegModel is designed for regression tasks only.")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.param_search_space = {
            "poly__degree": Integer(1, 5),  # Degree of polynomial features
            "regressor__alpha": Real(1e-4, 10.0, prior="log-uniform"),  # Regularization strength
            "regressor__l1_ratio": Real(0.0, 1.0),  # Mix between Lasso (1.0) and Ridge (0.0)
            "regressor__fit_intercept": Categorical([True, False]),  # Whether to calculate the intercept for this model
        }
        
        
    
    
    def explain_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimization first.")
        import matplotlib.pyplot as plt
        
        if not hasattr(self._best_model, "predict") or not callable(self._best_model.predict):
            raise RuntimeError("Best model does not have a predict method.")
        
        X, y = self.train_X, self.train_y
        
        model = copy.deepcopy(self._best_model)
        
        model.fit(X, y)
        
        inner_model = model.named_steps["model"] if hasattr(model, "named_steps") and "model" in model.named_steps else model
        regressor = inner_model.named_steps["regressor"]
        poly = inner_model.named_steps["poly"]

        # Get feature importances or coefficients
        if hasattr(regressor, 'coef_'):
            importances = regressor.coef_
        else:
            importances = regressor.feature_importances_
        
        base_features = X.columns if hasattr(X, "columns") else [f"X{i}" for i in range(X.shape[1])]
        try:
            if hasattr(model, "named_steps") and "model" in model.named_steps:
                base_features = model[:-1].get_feature_names_out()
            feat_names = poly.get_feature_names_out(base_features)
        except Exception:
            feat_names = [f"Feature {i}" for i in range(len(importances))]
        
        plt.figure(figsize=(10, 5))
        plt.barh(feat_names, importances, color='skyblue')
        plt.title("Feature Importance (Coefficients)")
        plt.xlabel("Coefficient Value")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.show()
                
