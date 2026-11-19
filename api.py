from .registry import create_model, normalize_model_name, normalize_task
from .result import TrainingResult


PRESETS = {
    "fast": {
        "n_trials": 10,
        "timeout": 180,
        "crossval": False,
    },
    "balanced": {
        "n_trials": 30,
        "timeout": 500,
        "crossval": False,
    },
    "accurate": {
        "n_trials": 60,
        "timeout": 1200,
        "crossval": True,
    },
}


def train(
    task,
    model,
    X,
    y,
    *,
    tune=False,
    preset="balanced",
    metric=None,
    metric_direction=None,
    metric_response=None,
    params=None,
    n_trials=None,
    timeout=None,
    crossval=None,
    cv_splitter=None,
    cv_groups=None,
    fit_best=True,
    **model_kwargs,
):
    normalized_task = normalize_task(task)
    preset_config = _preset_config(preset) if tune else PRESETS["balanced"]

    if isinstance(model, str):
        model_name = normalize_model_name(model)
        if "crossval" not in model_kwargs and (tune or crossval is not None or cv_splitter is not None):
            model_kwargs["crossval"] = (
                True if cv_splitter is not None
                else preset_config["crossval"] if crossval is None else crossval
            )
        wrapper = create_model(model_name, normalized_task, **model_kwargs)
    else:
        model_name = model.__class__.__name__
        wrapper = model
        if tune and crossval is not None:
            wrapper.opt_crossval = crossval

    if cv_splitter is not None:
        if not tune:
            raise ValueError("cv_splitter requires tune=True")
        wrapper.opt_crossval = True
        wrapper.cv = cv_splitter
        wrapper.cv_groups = cv_groups

    if metric is not None:
        wrapper.set_loss_function(metric, direction=metric_direction, response=metric_response)

    if tune:
        wrapper.optimize(
            X,
            y,
            n_trials=preset_config["n_trials"] if n_trials is None else n_trials,
            timeout=preset_config["timeout"] if timeout is None else timeout,
            fit_best=fit_best,
        )
    else:
        wrapper.fit(X, y, params=params)

    return TrainingResult(
        model=wrapper,
        task=normalized_task,
        model_name=model_name,
        tuned=tune,
        preset=preset if tune else None,
        metric=metric,
        metadata={
            "n_trials": preset_config["n_trials"] if tune and n_trials is None else n_trials,
            "timeout": preset_config["timeout"] if tune and timeout is None else timeout,
            "crossval": getattr(wrapper, "opt_crossval", None),
            "metric_direction": getattr(wrapper, "_score_direction", None),
            "metric_response": getattr(wrapper, "_score_needs_response", None),
        },
    )


def _preset_config(preset):
    key = str(preset).strip().lower()
    try:
        return PRESETS[key]
    except KeyError as exc:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{preset}'. Valid presets: {valid}.") from exc
