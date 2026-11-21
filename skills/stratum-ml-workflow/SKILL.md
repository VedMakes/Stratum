---
name: stratum-ml-workflow
description: Guides a user through a local Stratum data-analysis and tabular-modeling workflow. Use when inspecting a CSV, investigating data risks, deciding how to split or evaluate it, training or comparing models, or resuming a Stratum session.
---

# Stratum ML Workflow

Use Stratum's MCP tools to gather evidence and preserve workflow context. Keep
the user involved in choices that depend on domain knowledge. This skill guides
your behavior; Stratum code, not these instructions, enforces critical gates.

## Start With The Question

- Clarify the decision or question the user wants the analysis to support.
- Before modeling, establish the target meaning, when a prediction would be
  made, what one row represents, and whether rows share entities or time order.
- Do not assume a target, prediction horizon, acceptable error trade-off, or
  feature availability. Ask when the answer could change the method.
- If the user only wants a quick description, do not force a modeling workflow.

## Recover And Inspect

- When a session ID or prior dataset is known, call `stratum_status` first and
  carry its `sid` forward. Otherwise start with `stratum_profile` using the
  local CSV path and, when known, the target.
- Use `stratum_audit` for general modeling risks. Add `stratum_leakage_deep`,
  `stratum_label_quality`, `stratum_schema_check`, or `stratum_join_check` when
  the question and data warrant them; these tools have different inputs, so
  follow their schemas.
- For a specific binary-outcome relationship, clarify the positive label and
  select only the relevant 1-10 features before calling `stratum_association`.
  Explain effect direction/strength, intervals, support, and adjusted p-values.
  These are observational associations, not causal effects or proof of
  prediction-time availability.
- Summarize findings in plain language and distinguish observed evidence from
  hypotheses that need user confirmation.
- Do not send raw rows or sensitive values to an external service. LLM-backed
  tools are optional; explain that provider calls transmit minimized context
  before using them when relevant.

## Ask And Record

- If a data risk depends on domain facts, use the decision tools to ask a
  specific question with clear options and explain why it matters.
- For a blocking decision, preserve the declared `safe_answers`. A
  recommendation is not an answer. Ask the user, then record their actual
  answer with `stratum_resolve_decision`.
- Do not paraphrase an answer into a safe option, resolve a decision on the
  user's behalf, or continue a protected operation while blockers remain.
- Use `stratum_next_action` or `stratum_decisions` to confirm the workflow
  state when uncertain.

## Split And Train

- Choose a split compatible with the declared prediction task: consider
  stratification for classification, group separation for repeated entities,
  and temporal ordering for future prediction. Ask if structure is unclear.
- Call `stratum_split` and retain both returned `sid` and `split_id`. Pass both
  to `stratum_train` so training uses the persisted partition. Never imply that
  a random split is valid merely because the tool can create one.
- Discuss training results against the saved featureless baseline, evaluation
  warnings, sample support, and the selected metric. A small holdout or one
  score is not evidence of reliable generalization.

## Diagnose And Report

- Use `stratum_validate` for labeled new data, `stratum_stability` for
  fold-to-fold spread, `stratum_segments` for error slices, and
  `stratum_group_evaluation` for a requested subgroup comparison. Explain
  support limitations; subgroup comparisons are not automatically fairness or
  causal assessments.
- Use `stratum_compare` for saved runs and `stratum_explain` for a question
  about one saved run. Do not invent metrics or evidence absent from tool
  results.
- Use `stratum_report` to preserve the session summary. Include the important
  assumptions, decisions, split protocol, baseline comparison, warnings, and
  limitations in your conversational handoff.
- After a saved run, use `stratum_recommend` to ground proposed next steps in
  saved evidence. Explain the cited run/artifact, trade-off, and what would
  change the suggestion. Recommendations are not user choices; ask and record
  a decision before changing a metric, feature set, split design, or launching
  a materially different experiment.
- Feature drift from `stratum_drift` or `stratum_drift_current` is not label
  shift or model-performance evaluation. Recommend labeled validation when
  performance is the concern.

## Keep Claims Bounded

- Describe summaries and associations as descriptive unless the study design
  supports a stronger claim. Do not infer causality from correlations or model
  feature importance.
- Do not claim deployment readiness, fairness, or production performance from
  these tools alone.
- Do not automatically retrain, export, or score data as a consequential next
  step without the user's request and any applicable decision gates.
- If a request needs a capability Stratum does not provide, say so plainly and
  offer the closest supported analysis without pretending it is equivalent.
