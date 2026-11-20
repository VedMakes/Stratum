from dataclasses import dataclass
from importlib import import_module


TASK_ALIASES = {
    "classification": "classification",
    "classifier": "classification",
    "classify": "classification",
    "clf": "classification",
    "regression": "regression",
    "regressor": "regression",
    "regress": "regression",
    "reg": "regression",
}

MODE_BY_TASK = {
    "classification": "Classifier",
    "regression": "Regressor",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    module: str
    class_name: str
    tasks: tuple[str, ...]
    aliases: tuple[str, ...] = ()
    description: str = ""
    optional_extra: str | None = None

    def load_class(self):
        try:
            module = import_module(self.module, __package__)
        except ModuleNotFoundError as exc:
            if self.optional_extra and exc.name:
                raise ImportError(
                    f"Model '{self.name}' requires optional dependencies; "
                    f"install with `pip install 'mlproj[{self.optional_extra}]'`."
                ) from exc
            raise
        return getattr(module, self.class_name)


MODEL_SPECS = {
    "random_forest": ModelSpec(
        name="random_forest",
        module=".statistical.random_forest",
        class_name="RandomForestModel",
        tasks=("classification", "regression"),
        aliases=("rf", "forest"),
        description="Strong tabular baseline with feature importances.",
    ),
    "xgboost": ModelSpec(
        name="xgboost",
        module=".statistical.xgboost_model",
        class_name="XGBoostModel",
        tasks=("classification", "regression"),
        aliases=("xgb",),
        description="Gradient boosting for high-performing tabular models.",
        optional_extra="xgboost",
    ),
    "lightgbm": ModelSpec(
        name="lightgbm",
        module=".statistical.lightgbm_model",
        class_name="LightGBMModel",
        tasks=("classification", "regression"),
        aliases=("lgbm", "lgb"),
        description="Fast gradient boosting for larger tabular datasets.",
        optional_extra="lightgbm",
    ),
    "catboost": ModelSpec(
        name="catboost",
        module=".statistical.catboost_model",
        class_name="CatBoostModel",
        tasks=("classification", "regression"),
        aliases=("cat",),
        description="Gradient boosting with good categorical-feature handling.",
        optional_extra="catboost",
    ),
    "svm": ModelSpec(
        name="svm",
        module=".statistical.svm",
        class_name="SVMModel",
        tasks=("classification", "regression"),
        aliases=("svc", "svr"),
        description="Kernel SVM with scaling in the default pipeline.",
    ),
    "knn": ModelSpec(
        name="knn",
        module=".statistical.knn",
        class_name="KNNModel",
        tasks=("classification", "regression"),
        aliases=("nearest_neighbors", "k_neighbors"),
        description="Simple distance-based baseline.",
    ),
    "logistic_regression": ModelSpec(
        name="logistic_regression",
        module=".statistical.logreg",
        class_name="LogRegModel",
        tasks=("classification",),
        aliases=("logreg", "logistic", "lr"),
        description="Interpretable linear classifier.",
    ),
    "polynomial_regression": ModelSpec(
        name="polynomial_regression",
        module=".statistical.multireg",
        class_name="MultiRegModel",
        tasks=("regression",),
        aliases=("multireg", "multi_reg", "elastic_poly"),
        description="Polynomial features with ElasticNet regression.",
    ),
}

ALIASES = {
    alias: name
    for name, spec in MODEL_SPECS.items()
    for alias in (name, *spec.aliases)
}


def normalize_task(task):
    key = _normalize_key(task)
    try:
        return TASK_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(set(TASK_ALIASES.values())))
        raise ValueError(f"Unknown task '{task}'. Valid tasks: {valid}.") from exc


def normalize_model_name(name):
    key = _normalize_key(name)
    try:
        return ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unknown model '{name}'. Available models: {valid}.") from exc


def available_models(task=None):
    normalized_task = normalize_task(task) if task is not None else None
    names = []
    for name, spec in MODEL_SPECS.items():
        if normalized_task is None or normalized_task in spec.tasks:
            names.append(name)
    return names


def model_info(task=None):
    normalized_task = normalize_task(task) if task is not None else None
    info = {}
    for name, spec in MODEL_SPECS.items():
        if normalized_task is None or normalized_task in spec.tasks:
            info[name] = {
                "tasks": spec.tasks,
                "aliases": spec.aliases,
                "description": spec.description,
            }
    return info


def get_model_spec(name, task=None):
    model_name = normalize_model_name(name)
    spec = MODEL_SPECS[model_name]
    if task is not None:
        normalized_task = normalize_task(task)
        if normalized_task not in spec.tasks:
            tasks = ", ".join(spec.tasks)
            raise ValueError(f"Model '{model_name}' does not support {normalized_task}. It supports: {tasks}.")
    return spec


def create_model(name, task, **kwargs):
    normalized_task = normalize_task(task)
    spec = get_model_spec(name, normalized_task)
    model_cls = spec.load_class()
    return model_cls(mode=MODE_BY_TASK[normalized_task], **kwargs)


def _normalize_key(value):
    if value is None:
        raise ValueError("Value cannot be None.")
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")
