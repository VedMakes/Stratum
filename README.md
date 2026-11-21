# Stratum

Local-first tabular ML for AI agents.

Stratum is the main product here: an MCP server that lets an agent inspect CSVs,
audit modelling risks, split data, train models, validate them, score new rows,
track user decisions, and produce reports without handing raw data to a remote
service.

It ships with **MLProj**, a small modular statistical modelling library. Stratum
uses MLProj under the hood for model wrappers, tuning, scoring, feature
importance, and fitted artifacts; you can also use MLProj directly from Python.

## What You Get

- Agent-facing MCP tools for profiling, drift, leakage, label quality, splitting,
  training, validation, prediction, export, and reporting.
- A two-way decision layer so the agent can pause for domain judgement, record the
  answer, and continue with session context intact.
- Local persistence for sessions, runs, decisions, journals, models, and data
  contracts.
- Provider-neutral optional LLM support for semantic feature engineering and
  explanation; everything else runs locally.
- Direct Python access to the underlying MLProj modelling API.

## Install

Stratum requires Python 3.12 or newer.

```bash
git clone https://github.com/VedMakes/Stratum
cd Stratum
python -m venv .venv
. .venv/bin/activate
python -m pip install -e ".[stratum,test]"
python -m pytest
```

Smoke-check the MCP entrypoint:

```bash
stratum-mcp --help
stratum-mcp --version
```

For only the core ML library, install without extras:

```bash
python -m pip install -e .
```

The `stratum` extra installs the MCP adapter; the command-line workflow uses
the core dependencies. Optional model and analysis integrations are separate:

```bash
python -m pip install -e ".[catboost]"       # or .[lightgbm], .[xgboost]
python -m pip install -e ".[boosters]"       # all three optional boosters
python -m pip install -e ".[smote]"
python -m pip install -e ".[visualization]"
python -m pip install -e ".[stratum,llm-openai]"  # MCP plus OpenAI-backed tools
```

The built-in `auto` model selection only uses models available in the current
environment. An explicitly selected optional model reports which extra to
install when its dependency is missing.

## CLI Quickstart

The CLI provides direct local access to the same profiling, audit, split,
training, validation, reporting, and artifact workflows. This synthetic example
contains no real customer data and is intentionally tiny; its model metrics are
not evidence of useful generalization:

```bash
stratum profile examples/churn.csv --target churn
stratum audit examples/churn.csv --target churn --sid YOUR_SID
stratum split examples/churn.csv --target churn --mode stratified --sid YOUR_SID
```

Each command prints JSON. Reuse the returned `sid`; pass the split command's
`split_id` to training so the model uses exactly those saved partitions:

```bash
stratum train examples/churn.csv --target churn --model logistic_regression \\
  --task classification --sid YOUR_SID --split-id YOUR_SPLIT_ID
stratum validate YOUR_MODEL_ID examples/churn.csv --target churn --sid YOUR_SID
stratum report --sid YOUR_SID --format html --output session-report.html
stratum artifacts --sid YOUR_SID
```

For monitoring against the immutable training-time feature summary saved with
the model, use:

```bash
stratum drift-current YOUR_MODEL_ID new_customers.csv --sid YOUR_SID
```

Or compare two explicit CSV files with `stratum drift reference.csv current.csv`.
Both drift commands accept `--psi-mild-threshold`,
`--psi-significant-threshold`, `--psi-high-threshold`, and
`--p-value-threshold`. Drift reports describe feature-distribution change, not
target/label shift or model performance. P-values are unadjusted for multiple
comparisons, and drift findings are prompts to investigate and validate on
labeled current data, not automatic retraining decisions. Older runs without a
stored training-time profile require an explicit reference CSV.

## MCP Setup

After installation, configure your MCP client to launch `stratum-mcp`:

```json
{
  "mcpServers": {
    "stratum": {
      "command": "stratum-mcp"
    }
  }
}
```

The stdio setup is covered by a subprocess smoke test that completes MCP
initialization and lists the registered tools. CI runs this handshake against
the installed package.

If your client does not inherit the virtualenv `PATH`, use the absolute script
path instead, for example:

```json
{
  "mcpServers": {
    "stratum": {
      "command": "/path/to/Stratum/.venv/bin/stratum-mcp"
    }
  }
}
```

For isolated local runs:

```bash
export STRATUM_HOME=/tmp/stratum-home
export STRATUM_DB_PATH=/tmp/stratum-home/sessions.db
export MPLCONFIGDIR=/tmp/matplotlib
```

## Optional LLM Tools

Only `stratum_engineer` and `stratum_explain` use an LLM provider. All profiling,
auditing, splitting, training, validation, prediction, comparison, export, and
reporting tools are local.

Install a provider extra when you want LLM-backed feature ideas or narrative
explanations:

```bash
python -m pip install -e ".[llm-openai]"
# or
python -m pip install -e ".[llm-anthropic]"
```

Configure provider-neutral environment variables:

```bash
export STRATUM_LLM_PROVIDER=openai
export STRATUM_LLM_MODEL=your-model
export STRATUM_LLM_API_KEY=your-key
```

Provider-specific variables such as `OPENAI_API_KEY`, `OPENAI_MODEL`,
`ANTHROPIC_API_KEY`, and `ANTHROPIC_MODEL` are also accepted.

Configuring an LLM provider enables external transmission for only
`stratum_engineer` and `stratum_explain`. Engineering sends aggregate schema
statistics, anonymized target-class counts, the requested goal, prior aggregate
metrics and feature names, and minimized decision status. Explanation sends the
question, run metrics/parameters/features, and minimized decision status. Raw
rows, target labels, resolved-flag comments/reasoning, and data-derived error
trees are not included. Known sensitive column names, emails, SSNs, UUIDs, and
labeled identifiers are redacted, but heuristic redaction cannot recognize
every sensitive value. Review prompts before using an external provider and
follow that provider's data-handling policy. Session text is stored locally in
SQLite; on POSIX systems, the database file is restricted to the current OS user
(owner read/write).

## Agent Workflow

1. Start with `stratum_status` to recover prior runs, decisions, and current best
   model for the session.
2. Profile and contract the dataset with `stratum_profile`.
3. Audit data risks with `stratum_audit`, `stratum_leakage_deep`,
   `stratum_label_quality`, `stratum_join_check`, and `stratum_schema_check`.
4. Use `stratum_next_action`, `stratum_decisions`, `stratum_create_decision`, and
   `stratum_resolve_decision` when the agent needs user judgement.
5. Choose a split with `stratum_split` (`random`, `stratified`, `group`,
   `temporal`, or `predefined`; temporal splits also support a row gap), then
   check contamination with `stratum_contamination`. Group splits keep groups
   disjoint and exclude the group column from model features. Predefined splits
   use a train/test assignment column and remove it from the saved partitions.
6. Pass the returned `split_id` and `sid` to `stratum_train` to train and
   evaluate on those exact saved partitions. Each split has unique persisted
   files and a manifest with source and partition checksums, so later splits do
   not invalidate earlier split IDs. Training without a `split_id` retains a
   deterministic internal holdout and records its strategy. Tuning from group
   and temporal splits uses GroupKFold and TimeSeriesSplit respectively;
   predefined splits reject exact row-value overlap across partitions. Training
   compares against a majority-prior classifier or mean-prediction regressor on
   the same holdout, and returns per-class scores, probability metrics when
   supported, and evaluation warnings. Validate with
   `stratum_validate`, investigate error slices with `stratum_segments`, compare
   task metrics across a selected column with `stratum_group_evaluation`, and
   estimate fold spread with `stratum_stability`. Group evaluation reports
   support thresholds and is descriptive, not a fairness or causal assessment.
   Training defaults to accuracy for classification and RMSE for regression;
   pass `metric` and optionally `metric_direction` to state another tuning
   objective. The chosen objective is saved separately from holdout metrics.
   Run and compare outputs include local training time and warm batch inference
   latency as hardware-dependent context, not as a combined model score.
   Stability evaluation reuses the saved split design:
   repeated stratified/group folds for IID/group data, or ordered walk-forward
   folds with the saved gap for temporal data. Predefined partitions are not
   randomly re-split.
7. Score new data with `stratum_predict`, monitor drift with
   `stratum_drift_current`, compare runs with `stratum_compare`, export with
   `stratum_export`, and summarize the trail with `stratum_report`.

Blocking decisions must declare `safe_answers`. A recommendation is not a user
decision: only an answer explicitly listed as safe clears the workflow gate.
Unknown, deferred, and unsafe answers remain visible and keep training,
prediction, and export blocked until the decision is safely resolved.

## MCP Tool Map

Session and decision tools:

- `stratum_status`
- `stratum_next_action`
- `stratum_create_decision`
- `stratum_decisions`
- `stratum_resolve_decision`
- `stratum_log_event`
- `stratum_journal`

Data and guardrail tools:

- `stratum_profile`
- `stratum_schema_check`
- `stratum_split`
- `stratum_drift`
- `stratum_drift_current`
- `stratum_audit`
- `stratum_leakage_deep`
- `stratum_label_quality`
- `stratum_contamination`
- `stratum_join_check`

Modelling and analysis tools:

- `stratum_engineer`
- `stratum_train`
- `stratum_validate`
- `stratum_segments`
- `stratum_group_evaluation`
- `stratum_stability`
- `stratum_predict`
- `stratum_explain`
- `stratum_compare`
- `stratum_export`
- `stratum_report`

## Direct MLProj Usage

MLProj gives the local modelling layer a stable Python API. Use it directly when
you want ordinary script/notebook control without the MCP loop.

```python
from MLProj import train

result = train(
    task="classification",
    model="random_forest",
    X=X_train,
    y=y_train,
)

predictions = result.predict(X_test)
metrics = result.evaluate(X_test, y_test)
print(metrics)
```

Supported classification and regression models:

- `random_forest` aliases: `rf`, `forest`
- `xgboost` aliases: `xgb`
- `lightgbm` aliases: `lgbm`, `lgb`
- `catboost` aliases: `cat`
- `svm` aliases: `svc`, `svr`
- `knn` aliases: `nearest_neighbors`, `k_neighbors`

Classification only:

- `logistic_regression` aliases: `logreg`, `logistic`, `lr`

Regression only:

- `polynomial_regression` aliases: `multireg`, `multi_reg`, `elastic_poly`

## Tuning And Metrics

Set `tune=True` to use Optuna. Presets control trial count, timeout, and whether
cross-validation is used.

```python
result = train(
    task="regression",
    model="xgboost",
    X=X_train,
    y=y_train,
    tune=True,
    preset="balanced",
)

print(result.best_params)
print(result.score(X_test, y_test))
```

Available presets:

- `fast`: 10 trials, shorter timeout, no cross-validation
- `balanced`: 30 trials, moderate timeout, no cross-validation
- `accurate`: 60 trials, longer timeout, cross-validation enabled

You can override the preset:

```python
result = train(
    "classification",
    "lightgbm",
    X_train,
    y_train,
    tune=True,
    n_trials=100,
    timeout=1800,
    crossval=True,
    metric="f1",
)
```

`result.evaluate(X, y)` returns aggregate and per-class classification metrics.
When probabilities are available it also reports ROC-AUC, binary PR-AUC
(average precision, with the positive class identified in the metrics), and
Brier score. Regression evaluation reports MAE, RMSE, MSE, and R-squared.

Built-in metrics include `accuracy`, `f1`, and `roc_auc` for classification, and
`rmse`, `mse`, `mae`, and `r2` for regression. Custom metrics are supported by
passing a callable with `metric_direction="minimize"` or
`metric_direction="maximize"`.

For imbalanced classification, pass `use_smote=True`. Resampling runs inside
the model pipeline after fold-specific imputation and encoding, so CV validation
folds and holdout rows remain untouched. The final fitted pipeline resamples
only the data passed to its training fit.

## Privacy And Storage

Raw data and row-level values stay on the machine. The journal stores structured
session summaries and payloads, not full raw conversation transcripts. When
enabled, LLM calls use aggregate schema statistics, anonymized target-class
counts, the user's goal or question, saved metrics and feature names, and
minimized decision status. Raw target labels, resolved-flag comments/reasoning,
and data-derived error trees are excluded. Known sensitive column names and
common email, SSN, UUID, and labeled identifier patterns are redacted, but
heuristics cannot identify every sensitive value. Calls degrade to local results
when the provider, key, model, SDK, or API is unavailable.

By default, fitted models are stored under `~/.stratum/models` and sessions under
`~/.stratum/sessions.db`. Override both with `STRATUM_HOME`, or only the database
with `STRATUM_DB_PATH`.

Each trained run records a versioned provenance manifest: source CSV hash and
frame fingerprint, schema, target/features, split and selection protocol,
hyperparameters, preprocessing pipeline, decision IDs, environment versions,
and the fitted artifact checksum. Prediction and evaluation verify the
recorded artifact path and SHA-256 before deserialization; incompatible library
versions produce warnings. Older runs without verifiable provenance are not
loaded and must be retrained.

Joblib deserialization can execute code. Stratum only supports artifacts created
by its local training workflow; never register or load a `.joblib` file from an
unknown source. The checksum detects changes relative to the saved run record,
but is not a signature proving who created the artifact.

Artifacts are retained locally and are not automatically pruned. Back up
`STRATUM_HOME` and, when configured separately, `STRATUM_DB_PATH` together.
Deleting individual model files leaves runs that can no longer be loaded. For
full cleanup, stop Stratum and remove its home directory and any separately
configured session database after preserving anything you need.

## Project Layout

- `stratum/`: MCP orchestration, tools, sessions, privacy, reports, and artifacts.
- `api.py`: high-level MLProj `train()` facade.
- `base_model.py`: shared model lifecycle, scoring, tuning, and CUDA routing.
- `registry.py`: model names, aliases, and task support.
- `result.py`: returned training result object.
- `ensemble.py`: ensemble search and voting.
- `statistical/`: statistical and tabular model wrappers.
- `tests/`: regression tests for the MCP tools and MLProj integration.

## Testing

Run the suite with:

```bash
python -m pytest
```

For the broader optional-dependency test environment:

```bash
python -m pip install -e ".[stratum,test,boosters,smote,visualization]"
python -m pytest
```

Test caches, coverage files, JUnit XML, and local result folders are ignored by
Git. The `tests/` directory itself is intentionally tracked.
