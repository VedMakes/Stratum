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
  MCP SDK, while the in-process handshake passed. Treat this as an unresolved
  SDK/stdio transport investigation, not a Stratum-specific defect until
  isolated further.
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
- For targets with at most 20 distinct non-null values,
  `sanitise_for_api()` includes the raw target category labels in `value_counts`.
  Column names are redacted when PII-like, but categorical target values are
  not anonymized there.
- In MLProj, `_prepare_training_data()` applies SMOTE before
  `cross_val_score()` creates CV folds. This allows resampled observations to
  be distributed across folds. The same method can prompt through `input()` for
  imbalanced data, which is unsuitable for unattended MCP execution.
- `dispatch_tool()` calls synchronous handlers inline. Long training or tuning
  work can occupy the async server event loop.
- The source package `stratum/data/` and CI workflow are currently visible in
  the worktree but not committed. Installation tests included the local source
  package by overlaying it into the temporary clone.

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
  **68 passed** in 3.74 seconds.

Next linear task: M1-03, establish and test the privacy boundary for persisted
session context and any data sent to optional LLM providers, following the
roadmap.

