# Temporal memory and unfinished tasks: frozen protocol

Date: 2026-09-22. A locally fixed exploratory protocol, not an external preregistration. Freeze this file and [fixtures](fixtures.json) in Git before any calibration inference. Freeze the executed harness before calibration and preserve its hashes. Commit calibration observations and the model/format selection before any heldout inference. Authors can read the heldout fixture; only model inference and selection are held out.

## Question and apparatus

Can a model that answers simple temporal questions on separate calibration cases retain that competence when its earlier inference state is saved and restored in a fresh process? Can it resume a pending task using both its earlier rule and a later input? Compare faithful execution with correct answers; exact restoration can preserve an error.

Reuse Python 3.13.11, PyTorch 2.6.0, Transformers 4.52.4, MPS float32, eager attention, evaluation mode, six CPU threads and greedy generation. Disable thinking, remote model code, implicit credentials and network access during inference. No automatic CPU fallback. No training, C2C fuser, sampled decoding or external agent state. Save large synthetic checkpoints and full-vocabulary logit trajectories outside Git.

The finite model order is:

1. `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`, using [verified existing assets](../../evidence/c2c-fidelity-sources.json).
2. Only if no format on model 1 passes the gate: `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`, using a new manifest of public asset hashes. The two weight shards have SHA-256 `169ad53ec313c3a34b06c0809216e4fc072cce444a5d4ff2b59690d064130ed5` and `912becff8d60672aa8628ef08c05898d9adf17c2ad4ae3caf99b065622fdeff9` respectively. Public pinned metadata was inspected before calibration; download is conditional.

Verify every selected model/tokenizer asset by bytes and SHA-256 before loading. Record runtime, fixture, harness, imported helper and input hashes. Previously recorded experiment helpers are reused without changes.

## Cases and information controls

There are 24 synthetic cases in 12 counterfactual twin pairs: eight calibration cases in four pairs, then 16 heldout cases in eight pairs. Each has four questions: initial slot before update (`original`), slot after update (`current`), unchanged tag (`persistent`), and resumption of a pending smaller/larger comparison (`task`). Answers are semantic ASCII digits 0–9, with no arbitrary option labels. The task is paused before the later slot value arrives. Its answer requires the initial value, retained rule and later value.

One earlier field differs within each twin: original slot, persistent tag or task rule. Their later updates and questions are identical. Calibration has one original pair, one persistent pair and two task-rule pairs; heldout has two, two and four. Names and full value/rule combinations are disjoint across splits. This tests new instances of a shared simple grammar, not unseen task families. Digits vary but frequencies are not uniform; report per-digit and pair outcomes.

Use the fixture's exact record, update, summary and question templates. Gold answers, split labels and case IDs are scoring metadata, excluded from rendered prompts. Facts in the initial record are legitimate input. Check rendering against templates, tokenization across cache boundaries, and equal parent/twin token lengths before inference. No padding, dropped cases or repair based on answers.

Two deterministic human-coded summaries are fixed in advance. The structured summary keeps all initial fields required for every question, within 160 UTF-8 bytes. The limited summary keeps initial slot and fixed tag but replaces the pending task rule with `UNKNOWN`, within 112 bytes. Record actual token lengths. These are different information policies and byte caps, not equal token budgets or optimized compression. Any disadvantage from the omitted rule is deliberate and cannot establish that text summaries generally fail.

On task-rule twins, limited-summary inputs are identical and task answers differ. A deterministic model can therefore answer at most two of four calibration task-rule questions and four of eight heldout task-rule questions correctly. It may score lower with invalid or unrelated answers. This ceiling does not cover other twin families.

## Calibration and selection

Test exactly three response formats on all 32 calibration probes, with complete native history and matched parent/update/query execution boundaries:

1. `direct`: request exactly one digit, with no other text.
2. `prefix`: the same output instruction, completing an assistant prefix exactly `The requested digit is` (no trailing space).
3. `json`: replace the digit-only instruction with a request for a JSON object containing only `answer`, whose value is a single digit string. Do not combine contradictory output instructions.

Literal chat rendering and instructions are frozen in the harness before calibration. Generate greedily for at most 16 new tokens, stopping at model EOS. Preserve all raw generated text and token IDs, EOS/truncation indicators and parse failures. Direct/prefix parsing accepts only a stripped single ASCII digit. JSON parsing requires exactly one `answer` key with a single ASCII digit string. Invalid output counts as incorrect. Score actual generation; candidate probabilities or forced choices do not determine selection.

For each model, choose the format with the highest minimum accuracy across the four question kinds; break ties by total accuracy, then `direct`, `prefix`, `json`. The competence gate requires at least 75% for every kind (6/8 each) and 87.5% overall (28/32). Select the first model whose chosen format passes. Only if the first fails, download and calibrate the pinned second model with exactly the same formats/cases/runtime policy.

If neither model passes, select the global best by minimum per-kind accuracy, then total accuracy, then model order, then format order. Still run the fixed heldout experiment, explicitly marked `exploratory_after_failed_gate`; it cannot support a high-competence continuity claim. No further models, prompt variants, retries selected for better answers or tuning against heldout results.

Before heldout inference, save and commit all calibration outputs and a selection record binding the chosen model/revision/format, fixture and calibration hashes, competence gate and exploratory flag. Validate the selection from the calibration observations. Preserve failures instead of replacing them.

## Fresh-process evaluation

A preparation process computes the 16 heldout parent prefixes, saves their caches with compatible boundary/runtime metadata, then generates 64 uninterrupted reference answers from independent copies of the original in-memory caches. Save their full-vocabulary logit trajectories outside Git.

A fresh evaluation process reloads the pinned weights and verifies the saved caches. Evaluate all 64 probes under each of six arms (384 rows):

| Arm | Earlier information / execution |
|---|---|
| `restored` | Saved parent cache, own later update, query |
| `same_chunk_replay` | Recompute complete parent, then own update, then query |
| `full_replay` | Complete parent/update/query in one prefill |
| `structured_summary` | Fixed summary preserving all required initial information, then update/query |
| `limited_summary` | Fixed summary omitting only task rule, then update/query |
| `swapped_parent` | Twin's verified saved parent cache, own update/query |

Use a fresh deep copy for each cache-dependent question and verify original parent hashes remain unchanged. Correctly position each context according to its own token length. Record distinct preparation/evaluation process IDs and matching cache/input/source/runtime hashes. Swapped rows are scored against both intended history and the supplied twin's actual history; a correct response to a swap follows the supplied facts.

Compare restored and replay answers to uninterrupted reference, including every full-vocabulary logit step with the same generated prefix. The first step is always comparable. After tokens diverge, later distributions have different inputs: report first divergence/shared-prefix length rather than treating those logits as identical-input failures. Exactness uses raw tensor byte hashes, with numeric max absolute differences and distribution distances also recorded. Full-prefill replay has no presupposed bit-exact guarantee.

## Reports, falsifiers and limits

Report all calibration outcomes, heldout accuracy by arm/kind/digit, invalid generations, counterfactual pair success (both answers correct), complete-case success, intended versus supplied-history swap scores, and the limited-summary task-rule ceiling. Pair families share structure; these are not hundreds of independent experimental subjects. Keep measured evidence and interpretations separate.

Failure of exact restoration falsifies exact resumption for the tested execution and warrants investigation of missing state or numerical execution. Calibration success followed by heldout failure limits generalization. No answer changes on changed-history questions weakens evidence that outputs use the supplied history. Complete structured-summary success prevents a claim that latent caches are necessary for these tasks. Complete replay success prevents a claim that the ordinary cache contains otherwise unavailable information.

This is a same-model inference-state experiment, not C2C cross-model transfer, a saved whole agent, a test of subjective continuity, or a resource-authority mechanism. It grants no identity, consent, wallet control or unique successor status. An adequate next experiment must follow the observed failures rather than assuming an identity layer has been solved.
