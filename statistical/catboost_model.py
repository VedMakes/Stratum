from ..base_model import BaseModel
from catboost import CatBoostClassifier, CatBoostRegressor
from skopt.space import Integer, Real, Categorical
import catboost as cb


class CatBoostModel(BaseModel):
    def __init__(self, mode=None, crossval=False, use_smote=False, supports_cuda=False, verbose=0):
        super().__init__(mode = mode, crossval = crossval, needs_scaling = False, use_smote= use_smote, supports_cuda = supports_cuda, verbose=verbose)
        
        if mode == "Regressor":
            self._model = CatBoostRegressor(
                random_state=42,
                task_type='GPU' if self._enable_CUDA else 'CPU',
                verbose=False,
                allow_writing_files=False,
            )
        elif mode == "Classifier":
            self._model = CatBoostClassifier(
                random_state=42,
                task_type='GPU' if self._enable_CUDA else 'CPU',
                verbose=False,
                allow_writing_files=False,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        
        self.param_search_space = {
            "iterations": Integer(100, 1000),
            "depth": Integer(1, 10),
            "learning_rate": Real(1e-3, 0.3, prior='log-uniform'),
            "l2_leaf_reg": Real(1e-4, 10.0, prior='log-uniform'),
            "subsample": Real(0.5, 1.0),
            "colsample_bylevel": Real(0.5, 1.0),
            "bootstrap_type": Categorical(["Bernoulli", "MVS"]),
            "random_strength": Real(1e-4, 10.0, prior='log-uniform'),
            "leaf_estimation_iterations": Integer(1, 10),
            "leaf_estimation_method": Categorical(["Newton", "Gradient"]),
        }
        if mode == "Classifier":
            self.param_search_space["scale_pos_weight"] = Real(0.5, 10.0)
        
        self.cv_method = cb.cv if self.opt_crossval else None
        

        
        
    def explain_model(self):
        if self._best_model is None:
            raise RuntimeError("No best model found. Run optimization first.")
        import shap
        
        if not hasattr(self._best_model, "predict") or not callable(self._best_model.predict):
            raise RuntimeError("model does not support prediction. Ensure it is fitted before SHAP explanation.")
        
        X, y = self.train_X, self.train_y
        estimator = self.final_estimator()
        transformed_X = self.transform_features(X)

        explainer = shap.Explainer(estimator)
        shap_values = explainer(transformed_X)

        shap.summary_plot(shap_values, transformed_X)
        shap.dependence_plot(0, shap_values, transformed_X)
        
        print("SHAP values calculated for the best model.")
        return shap_values
    
            
