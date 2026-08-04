from ..base_model import BaseModel
from xgboost import XGBRegressor, XGBClassifier
from skopt.space import Integer, Real
import xgboost as xgb


class XGBoostModel(BaseModel):
    def __init__(self, mode =  None, crossval = False, use_smote = False, supports_cuda=False, verbose=0):
        super().__init__(mode = mode, crossval = crossval, needs_scaling = False, use_smote= use_smote, supports_cuda = supports_cuda, verbose=verbose)
        
        if mode == "Regressor":
            self._model = XGBRegressor(random_state=42, enable_categorical=True, device='cuda' if self._enable_CUDA else 'cpu', tree_method = 'hist')
        elif mode == "Classifier":
            self._model = XGBClassifier(random_state=42, enable_categorical=True, device='cuda' if self._enable_CUDA else 'cpu', tree_method = 'hist')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.param_search_space = {
            "n_estimators": Integer(100, 1000),          # Number of trees
            "max_depth": Integer(3, 15),                 # Tree depth
            "learning_rate": Real(1e-3, 0.3, prior = 'log-uniform'),  # Step size shrinkage
            "subsample": Real(0.3, 1.0),                 # Row sampling
            "colsample_bytree": Real(0.3, 1.0),          # Column sampling
            "colsample_bylevel": Real(0.3, 1.0),         # Column sampling per tree level
            "colsample_bynode": Real(0.3, 1.0),          # Column sampling per node
            "reg_alpha": Real(1e-4, 10.0, prior = 'log-uniform'),     # L1 regularization
            "reg_lambda": Real(1e-4, 10.0, prior = 'log-uniform'),    # L2 regularization
            "gamma": Real(0.0, 20.0),                    # Minimum loss reduction
            "min_child_weight": Integer(1, 20),          # Minimum sum of instance weight
        }
        if mode == "Classifier":
            self.param_search_space.update({
                "max_delta_step": Integer(0, 10),
                "scale_pos_weight": Real(0.5, 10.0),
            })
        
        self.cv_method = xgb.cv if self.opt_crossval else None
        
        
    
    def _custom_cross_val_score(self, X, y, params, cv_splits):
        if self._enable_CUDA:
            return super()._custom_cross_val_score(X, y, params, cv_splits)
        self._ensure_score_spec()
        return self._cpu_cross_val_score(X, y, params, cv_splits)
        
        
    
    
    def explain_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimization first.")
        import shap
        
        if not hasattr(self._best_model, "predict") or not callable(self._best_model.predict):
            raise RuntimeError("model does not support prediction. Ensure it is fitted before SHAP explanation.")

        X, y = self.train_X, self.train_y
        estimator = self.final_estimator()
        transformed_X = self.transform_features(X)
        
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(transformed_X, check_additivity=False)

        shap.summary_plot(shap_values, transformed_X)
        shap.dependence_plot(0, shap_values, transformed_X)

        print("SHAP values calculated for the best model.")
        return shap_values

        
        
    
    
    
