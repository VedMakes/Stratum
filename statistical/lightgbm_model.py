from ..base_model import BaseModel
from lightgbm import LGBMRegressor, LGBMClassifier
from skopt.space import Integer, Real, Categorical
import lightgbm as lgb

class LightGBMModel(BaseModel):
    def __init__(self, mode = None, crossval = False, use_smote = False, supports_cuda=False, verbose=0):
        super().__init__(mode = mode, crossval = crossval, needs_scaling = False, use_smote= use_smote, supports_cuda = supports_cuda, verbose=verbose)
        
        if mode == "Regressor":
            self._model = LGBMRegressor(random_state=42, device ='gpu' if self._enable_CUDA else 'cpu')
        elif mode == "Classifier":
            self._model = LGBMClassifier(random_state=42, device='gpu' if self._enable_CUDA else 'cpu')
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.param_search_space = {
            "n_estimators": Integer(100, 1000),
            "max_depth": Integer(-1, 15),  # -1 means no limit
            "learning_rate": Real(1e-3, 0.3, prior='log-uniform'),  # Step size shrinkage
            "num_leaves": Integer(20, 300),  # More leaves = more complex
            "min_child_samples": Integer(5, 100),  # Minimum data in leaf
            "subsample": Real(0.5, 1.0),  # Bagging fraction
            "subsample_freq": Integer(1, 10),  # Bagging every k iterations
            "colsample_bytree": Real(0.5, 1.0),  # Feature fraction
            "reg_alpha": Real(1e-4, 10.0, prior='log-uniform'),  # L1 regularization
            "reg_lambda": Real(1e-4, 10.0, prior='log-uniform'),  # L2 regularization
            "boosting_type": Categorical(["gbdt", "dart", "goss"]),
        }
        if mode == "Classifier":
            self.param_search_space["scale_pos_weight"] = Real(0.5, 10.0)
        
        self.cv_method = lgb.cv if self.opt_crossval else None
    
    
    

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

        explainer = shap.Explainer(self._best_model)
        shap_values = explainer(X)
        
        shap.summary_plot(shap_values, X)
        shap.dependence_plot(0, shap_values.values, X)
        
        print("SHAP values calculated for the best model.")
        return shap_values
    
    
    
    
 
