# Stratum Engineering Baseline

Baseline date: 2026-11-19  
Roadmap: `Stratum_Engineering_Roadmap_Agent_Guideline.docx` (local-only source of truth)

This note records repository evidence and reproducible checks against the
engineering roadmap. It does not treat static code observations as runtime
failures unless a test or experiment demonstrated the behavior.

## Verification Method

- Python 3.12.3 on Linux.
- Created a temporary local clone of the current Git commit, then overlaid the
  working-tree `.gitignore` correction and the locally present `stratum/data/`
  source package. This models the tree after those changes are committed; it is
  not a clone of the remote GitHub repository.
- Installed with the README command: `python -m pip install -e ".[test]"`.
- Ran dependency consistency, import/registration, CLI, and pytest checks in the
  disposable environment.
- Did not run GitHub Actions remotely.

## Verified Results

- `stratum/data/` exists locally and contains `__init__.py`, `contracts.py`,
  `drift.py`, `profiler.py`, and `splitter.py`. These files were absent from the
  tracked tree because the unanchored `.gitignore` rule `data/` ignored nested
  source directories. Changing it to `/data/` makes the package visible. The
  files remain unstaged and must be included in a future commit.
- The roadmap document is ignored by its exact filename and does not appear as
  an untracked Git file.
- The documented editable install completed in the disposable Python 3.12.3
  environment. `pip check` reported no broken requirements.
- The clean install resolved `mcp==1.30.0` and downloaded
  `nvidia-nccl-cu13` (206.6 MB) through the default XGBoost dependency path.
  This is a verified installation-size cost; dependency splitting belongs to
  the operational-usability milestone.
- Core imports succeeded. `create_server()` constructed successfully, and all
  27 tool schemas had corresponding handlers.
- `stratum-mcp --help` and `stratum-mcp --version` succeeded; version output was
  `0.1.0`.
- In-process MCP initialization and `tools/list` succeeded with 27 tools.
- A subprocess stdio MCP initialization timed out in the clean environment. The
  same timeout reproduced with a minimal `mcp.Server` using the same installed
  MCP SDK, while the in-process handshake passed. This was later isolated and
  resolved in M5 below.
- At the M0 checkpoint, `python -m pytest` passed: **50 passed** in the
  disposable environment. The current working tree has additional focused and
  full-suite results recorded below.
- The new `.github/workflows/ci.yml` parses as YAML, and its equivalent import,
  registration, CLI, dependency, and test steps passed locally. The workflow
  itself has not yet run on GitHub Actions.

## Architecture Observed In Source

- The MCP adapter and dispatch live in `stratum/mcp_server.py`; tool schemas and
  implementations live in `stratum/schemas.py` and `stratum/tools/`.
- Data profiling, contracts, drift, and split logic live in `stratum/data/`.
- Session, decision, journal, run, and artifact persistence use SQLite in
  `stratum/session.py`; model files are persisted separately under the Stratum
  home directory.
- `stratum/core.py` resolves the root MLProj `train()` facade and training
  policy. Model wrappers and training behavior live in the shared root modules
  and `statistical/`.
- Prediction loads a saved joblib artifact and uses recorded feature names.
  Export generates a standalone retraining script.
- LLM-backed engineering uses `sanitise_for_api()` and `redact_context()`;
  explanation sends saved run evidence after name redaction. Other inspected
  workflow tools do not require an LLM provider.

## Pre-M1 Risks Confirmed By Source Inspection

These are code-path observations; their impact should be covered by focused
regression tests before being described as runtime failures.

- At the M0 baseline, `run_train()` did not query blocking decisions.
  M1-01 below closes that direct-call bypass.
- `run_train()` always creates a random 80/20 holdout with seed 42. The separate
  split tool supports `random`, `stratified`, and `temporal`, but training does
  not consume its saved split artifacts or record the selected split strategy.
  Group and predefined splits are not currently exposed by that tool.
- At the M0 baseline, decision rows had only `pending` and `resolved` statuses;
  resolution only checked membership in options. M1-01 adds a safe-answer
  contract and derived resolution states while preserving stored status values.
- At the M0 baseline, `sanitise_for_api()` included raw target labels and
  outbound contexts included data-derived error trees and free-text decision
  reasoning/comments. M1-03 below minimizes and tests these payloads.
- At the M0 baseline, MLProj applied SMOTE before CV folds were created. The
  fold-local sampler fix below closes that resampling leakage path. The legacy
  optional interactive prompt remains unsuitable for unattended callers when
  `use_smote=None` is explicitly selected.
- `dispatch_tool()` calls synchronous handlers inline. Long training or tuning
  work can occupy the async server event loop.
- The source package and CI workflow were verified in the working tree and are
  now committed in the repository history.

## Next Linear Tasks

M1-01 is implemented in the current worktree:

- Blocking decisions now require explicit `safe_answers`, validated against
  their options. Unknown, deferred, and rejected answers cannot be declared
  safe.
- Resolution retains the existing persisted `status` values for compatibility
  and exposes `resolution_state` (`pending`, `confirmed`, `rejected`, `unknown`,
  `deferred`, `unsafe`, or `invalidated`) plus `safe_to_proceed`.
- Training, prediction, and export enforce the session gate directly. The
  `next_action`, status, and report surfaces use the same blocker calculation.
- Temporal-availability decisions only treat `available_before_prediction` as
  safe. SQLite adds `safe_answers` through a backward-compatible schema
  migration.
- Focused regression tests passed for pending and unknown outcomes, direct tool
  calls, and the migration.

M1-02 is implemented in the current worktree:

- Split modes include deterministic random/stratified, temporal with an
  optional row gap, group-disjoint, and predefined assignment-column splits.
- Split manifests persist strategy parameters, source and partition checksums,
  row fingerprints, row counts, and excluded features. Each split gets unique
  output filenames, so a later split cannot overwrite an earlier split ID.
- Group-split tuning passes group labels to the existing MLProj CV API and uses
  `GroupKFold`; temporal-split tuning uses `TimeSeriesSplit`. Predefined splits
  reject exact row-value overlap across train and test before persistence.
- Training accepts `split_id`, verifies source and partition checksums and
  schemas, then evaluates on exactly the persisted partitions. Group identifiers
  are excluded from model features. The default internal holdout remains
  deterministic and its split strategy is recorded with each run.
- Session storage migrates existing run tables to include split metadata, and
  status exposes each run's split ID and strategy.
- Focused split, training, schema, session, and status tests passed. See the
  latest full-suite verification below.
- Current working-tree verification: `.venv/bin/python -m pytest -q` passed:
  **73 passed** in 4.91 seconds.

M1-03 privacy boundary is implemented in the current worktree:

- Target-class counts sent for feature engineering use anonymous `class_N`
  labels. Sensitive column names and common email, SSN, UUID, and labeled-ID
  patterns are redacted from string context.
- LLM contexts omit raw resolved-flag reasoning/comments and data-derived error
  trees; prior-run context is limited to aggregate metrics and feature names.
  Explanation questions receive the same known-pattern redaction.
- SQLite session database files receive owner-only (`0600`) permissions on POSIX.
- README documents which minimized fields leave the machine when a provider is
  configured and warns that heuristic redaction cannot identify every secret.
- Focused privacy and session permission tests pass; current full-suite result
  is recorded after final verification.
- Current working-tree verification: `.venv/bin/python -m pytest -q` passed:
  **73 passed** in 4.75 seconds.

Milestone 1 resampling fix is implemented:

- MLProj keeps original rows intact through split planning and evaluation.
  When enabled, an imbalanced-learn pipeline performs imputation and encoding,
  then SMOTE, then scaling/model fitting. During CV, each sampler is fit only on
  that fold's training indices; validation folds remain untouched.
- The final fitted pipeline samples only its training input, and its sampler is
  skipped for prediction and feature transformation. Feature importance retains
  encoded feature names.
- Regression coverage records row IDs seen by each fold-local sampler, checks
  an actual categorical-data SMOTE fit/predict path, and verifies feature names.
- Current working-tree verification: `.venv/bin/python -m pytest -q` passed:
  **75 passed** in 5.06 seconds.

Milestone 1 end-to-end workflow coverage is implemented:

- A single MCP-dispatch test covers profile/status, split persistence, an
  unresolved target decision, an `unknown` answer that remains blocking, safe
  resolution, exact-split training, validation, segment analysis, a second
  decision, and direct train/predict/export bypass attempts.
- The second blocker prevents all three operations and prevents prediction or
  export files from being written. After safe resolution, prediction, export,
  status, and report all reflect the completed workflow.
- Current working-tree verification: `.venv/bin/python -m pytest -q` passed:
  **76 passed** in 5.48 seconds.

Milestone 2 baseline comparison and task metrics are implemented:

- Training compares each fitted model against a majority-prior `DummyClassifier`
  or mean `DummyRegressor`, fit only on the training partition and evaluated on
  the same untouched holdout. Results record the baseline metrics and a
  direction-aware improvement score.
- Classification reports per-class precision/recall/F1/support, balanced
  accuracy, positive-class identity, ROC-AUC when defined, binary average
  precision (PR-AUC), and Brier score when probabilities are available.
  Regression reports MAE, MSE, RMSE, and R-squared.
- Evaluation warnings surface small holdouts, rare or missing classes, and
  substantial training imbalance. Run records persist the evaluation summary;
  status/report expose baseline comparison and warnings. Run comparison ignores
  nested per-class summaries when calculating scalar metric deltas.
- Full-suite verification: `.venv/bin/python -m pytest -q` passed:
  **79 passed** in 5.88 seconds.

Milestone 2 split-aware uncertainty evaluation is implemented:

- IID runs use repeated stratified or ordinary K-fold. Group runs use repeated
  shuffled `GroupKFold` with the persisted group column and are checked for
  group disjointness. Temporal runs use ordered `TimeSeriesSplit` with the
  persisted row gap; repeats are not misrepresented as independent temporal
  experiments. Predefined runs are directed to validation on their designated
  test partition instead of being randomly re-split.
- Results persist the method, folds/repeats, seed, metric, time/group parameters,
  source SHA256, data fingerprint, fold scores, repeat means, standard
  deviation, and descriptive 90% score range. The response explicitly warns
  that overlapping CV folds do not form independent confidence intervals.
- The MCP schema exposes repeat and seed controls; reports include the saved
  stability artifact protocol.
- Full-suite verification: `.venv/bin/python -m pytest -q` passed:
  **84 passed** in 4.50 seconds.

Milestone 2 support-aware subgroup evaluation is implemented:

- `stratum_group_evaluation` reports the shared task metrics overall and for
  each value of a user-selected column. Missing subgroup values form an
  explicit group; groups below the configurable support floor are excluded and
  counted in warnings.
- The tool stores a `group_evaluation` artifact, appears in session reports,
  and is registered in the MCP schema and handler map. Its response states that
  the comparison is descriptive, not a fairness or causal assessment.
- The focused diagnostics/schema/dispatch tests passed, and full-suite
  verification with `.venv/bin/python -m pytest -q` passed:
  **88 passed** in 5.82 seconds.

Milestone 2 methodological evaluation is complete against the roadmap:

- Task-appropriate baselines, per-class/probability classification metrics,
  regression metrics, robust residual summaries, calibration bins, and
  metric-suitability warnings are available in training/validation results.
- Repeated, split-aware stability evaluation and time-aware/group-aware
  evaluation preserve their saved protocols. Predefined holdouts are not
  randomly re-split. Group evaluation exposes support and excluded groups.
- The optimization metric and direction are configurable and persisted;
  session ranking and pairwise comparison honor the saved objective. Runs
  report warm local inference latency and training wall time alongside feature
  and saved stability summaries, without combining trade-offs into a score.
  Monetary cost is not estimated; timing is hardware/load dependent.
- Focused tests passed after each implementation increment. Final full-suite
  verification with `.venv/bin/python -m pytest -q` passed:
  **93 passed** in 4.83 seconds.

Milestone 3 reproducibility and artifact integrity is implemented:

- Run records have a backward-compatible provenance column containing a
  versioned source hash/frame fingerprint/schema, target/features/split,
  selection and metric definitions, hyperparameters, fitted pipeline,
  dependency/Python environment, artifact digest, and decision lineage.
- Split manifests and artifact metadata are versioned. Model loading in
  prediction and evaluation verifies the recorded path, artifact schema, and
  SHA-256 before joblib deserialization. Legacy runs without verifiable
  provenance and changed artifacts are refused; dependency-version drift is
  surfaced as a warning.
- The README defines the joblib trust boundary, legacy behavior, backup needs,
  and retention policy. Artifacts are retained without automatic pruning.
- Focused tests passed after each submodule. Final M3 full-suite verification
  with `.venv/bin/python -m pytest -q` passed: **99 passed** in 6.65 seconds.

Milestone 4 operational usability is implemented:

- Core dependencies are separated from MCP, boosters, SMOTE, visualization,
  LLM-provider, and test extras. Optional-model errors identify the relevant
  extra. CI installs the documented MCP/test environment and smoke-checks both
  command entrypoints.
- The `stratum` CLI covers profile, audit, split, train, validate, drift against
  explicit files or a model's immutable training-time feature profile, HTML/JSON
  reports, and artifact inspection. The HTML report is self-contained and
  includes model comparison, recorded decisions/risks, evaluation evidence,
  provenance, and recommended follow-up actions.
- Training provenance retains numeric quantile-bin counts and hashed
  categorical summaries for training features, without raw rows or raw
  category strings. Monitoring supports configurable PSI/p-value thresholds,
  severity and recommended actions, and persists findings. Legacy runs can use
  an explicit reference file. Monitoring assesses feature distributions only;
  label shift/performance needs labeled evaluation, and p-values are not
  corrected for multiple comparisons.
- The single README documents clean installs, both MLProj and Stratum entry
  points, optional extras, CLI workflows, privacy/trust boundaries, and drift
  limitations. `examples/churn.csv` is a small synthetic workflow fixture; its
  metrics are not a generalization claim.
- Focused drift/provenance/schema tests passed (45); focused CLI/packaging tests
  passed (9). A CLI smoke workflow completed profile, stratified split, model
  training, and HTML report generation. Final full-suite verification is
  recorded below after completion.

M0-M4 from the source roadmap are complete. The requested M5 transport
reliability increment is implemented:

- The timeout reproduced with the MCP SDK's default stdio reader and with a
  minimal SDK server. The SDK wrapped stdin in an AnyIO worker-thread file read;
  in this Python 3.12 environment it did not deliver the subprocess client's
  JSON-RPC input to the server.
- Stratum continues to use the SDK's stdio transport and protocol/session
  handling, but supplies an async line source backed by one daemon stdin-reader
  thread and a thread-safe queue. This avoids relying on the stalled worker-pool
  stdin read while keeping MCP framing and response handling in the SDK.
- A regression test launches the server as a subprocess with the official SDK
  client, completes `initialize`, calls `tools/list`, and checks that every
  registered schema is returned. The README and CI now cover the supported
  startup path.
- The documented `.[stratum,test]` install completed in a clean temporary
  environment with MCP 1.30.0 and AnyIO 4.16.0; `pip check` found no broken
  requirements. The installed `stratum-mcp` executable completed the same
  initialize/list-tools handshake.
- Focused MCP tests passed: **4 passed**. Final full-suite verification with
  `.venv/bin/python -m pytest -q` passed: **111 passed** in 18.41 seconds.

