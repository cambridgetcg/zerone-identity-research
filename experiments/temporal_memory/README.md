# Temporal memory and unfinished tasks

Date: 2026-09-22. **Completed exploratory experiment after a failed competence gate.** Native restoration reproduced all 64 reference trajectories exactly, including incorrect task answers. All 48 factual answers survived; pending-task accuracy was 6/16 with the original cache and 10/16 with a complete structured summary. Exact computational preservation and useful task continuation are different outcomes.

This follows the [checkpoint-branch study](../checkpoint_branches/README.md). It does not test subjective identity or cross-model C2C translation.

## Heldout results

The selected pinned Qwen3-1.7B generated 64 uninterrupted reference answers and 384 answers across six arms in separate preparation/evaluation processes. Every heldout answer satisfied the frozen JSON parser.

| Condition | Facts /48 | Pending tasks /16 | Total /64 |
|---|---:|---:|---:|
| Uninterrupted reference | 48 | 6 | 54 |
| Restored native checkpoint | 48 | 6 | 54 |
| Complete replay, matched chunks | 48 | 6 | 54 |
| Complete replay, one prefill | 48 | 6 | 54 |
| Complete structured summary | 48 | 10 | 58 |
| Limited summary, task rule omitted | 48 | 4 | 52 |
| Swapped checkpoint, scored against supplied history | 48 | 6 | 54 |

Restoration and matched-chunk replay each reproduced **64/64 token trajectories and 448/448 full-vocabulary logit steps byte for byte**. Complete replay in one prefill produced the same 64 token trajectories with small numerical differences: maximum absolute logit error `0.00026702880859375`, maximum total variation distance `0.00003677277437029989`. These are measurements for this fixed runtime, precision, model and short context.

Swapped caches produced exactly the corresponding twin's reference computation on all 64 questions. Against the intended original history they scored only 43/64 (40/48 facts and 3/16 tasks); against the history actually supplied they scored 54/64. This supports causal dependence on the saved history. Lower intended-history accuracy after a swap is not itself a reasoning error.

The structured summary preserved all required fields in a shorter representation and improved pending-task answers by four. It solved both task questions in 3/8 twin pairs, compared with 0/8 for the uninterrupted/restored condition. Complete-case success was 10/16 versus 6/16. These small paired observations do not establish a general summary advantage or an optimized compression policy. They do show that preserving this exact cache was unnecessary for retaining these facts, and that exact restoration did not repair task-use errors.

On the limited summary's four task-rule twin pairs, all eight task answers were the same digit, `3`: 1/8 correct, below the deterministic distinguishability ceiling of 4/8. All four identical-input pairs had identical full trajectories. This is a deliberate missing-information control, not a test of the best text memory under a budget.

All scores, per-digit counts, invalid-output counts, complete-case results and pair outcomes are in the [recomputed summary](../../evidence/temporal-memory-summary.json). [Raw observations](../../evidence/temporal-memory-results.json), [selection](../../evidence/temporal-memory-selection.json), [source and run receipt](../../evidence/temporal-memory-receipt.json) and [execution log](../../evidence/temporal-memory-run.log) preserve the evidence. Large local cache/logit files are represented by byte receipts, not committed to Git.

## Design

The [protocol](PROTOCOL.md) and [24-case fixture](fixtures.json) were frozen in commit `fcb09bab9e8a40a8acc1728faccfc83661164fdb`, followed by the inference harness in commit `1e5a32d`. Eight calibration cases choose among three fixed output formats; 16 heldout cases then compare six arms, with four questions per case. The heldout split contains new entities and value combinations within the same grammar. Authors can read the fixture; it is held out from inference and selection.

Each initial record contains an initial slot, an unchanged tag and a pending rule: once a later slot update arrives, choose the smaller or larger of the initial and updated values. The checkpoint is saved before that update. Questions ask about the original value, current value, persistent tag and resumed task. Counterfactual twins change exactly one earlier fact or rule while keeping the later update and questions identical.

Here, the unfinished task is a stored instruction awaiting a new input. We do not save a partly generated chain of reasoning, a learned policy update or an orchestrator's active task queue. Those would require additional state and separate tests.

The arms are native restoration, complete replay with matched execution boundaries, complete replay in one prefill, a structured summary preserving every required initial field, a limited summary omitting the task rule, and the twin's saved cache. Swaps are scored against both intended and actually supplied history. All generation is greedy, bounded at 16 new tokens and scored with the frozen strict parser.

The summaries are fixed human-coded information policies. Their 112/160-byte caps are not equal token budgets. A failure after deliberately omitting a rule cannot establish that text memory generally fails. Complete structured summary and complete replay are essential controls.

Including the chat opening but excluding the later update and question, heldout parent prefixes contain 57–59 tokens, structured summaries 31–33 and limited summaries 24–26. These are short, low-entropy synthetic records; long-context or long-term episodic memory is untested.

## Calibration

The competence gate requires at least 6/8 correct for every question kind and 28/32 overall. Model order is the existing pinned Qwen3-0.6B, followed only after failure by pinned Qwen3-1.7B. Rank formats by worst-kind accuracy, total accuracy and fixed format order. If both models fail, the best fixed candidate still enters evaluation, explicitly marked exploratory after a failed gate.

Both models failed the competence gate. The frozen global rule selected **Qwen3-1.7B with JSON, explicitly exploratory after a failed gate**. Selection and both calibrations were committed and pushed in `9e42ca7` before heldout inference. The independent standard-library analyzer verified the complete model/format policy before execution.

| Model | Format | Original /8 | Current /8 | Persistent /8 | Task /8 | Total /32 | Valid format /32 |
|---|---|---:|---:|---:|---:|---:|---:|
| 0.6B | Direct | 0 | 1 | 2 | 0 | 3 | 32 |
| 0.6B | Prefix | 0 | 0 | 0 | 0 | 0 | 0 |
| 0.6B | JSON | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7B | Direct | 8 | 6 | 3 | 0 | 17 | 26 |
| 1.7B | Prefix | 0 | 0 | 0 | 0 | 0 | 0 |
| 1.7B | JSON | 8 | 8 | 8 | 4 | 28 | 32 |

On the smaller model, direct generation returned `0` on 31/32 questions. Prefix responses added a period; JSON responses used Markdown fences. Both scored zero under the declared parser. A strict format score combines answer correctness with output compliance; it does not imply the model lacked every fact that appeared in an invalidly formatted response. No parser relaxation or additional format search was used.

The selected larger-model JSON condition produced valid output on all calibration questions. Its four task errors were wrong digits, so its failed gate cannot be explained by formatting alone. In the task-rule twins it repeated the same answer despite the reversed earlier rule. Excellent factual recall on this small calibration set did not establish reliable use of a pending instruction.

The full calibration records are retained for [0.6B](../../evidence/temporal-memory-calibration-0.6b.json) and [1.7B](../../evidence/temporal-memory-calibration-1.7b.json). The latter's [public asset manifest](../../evidence/temporal-memory-1.7b-sources.json) pins all ten downloaded files. No models beyond the declared pair or prompts beyond the three fixed formats were tried.

## Interpretation and next question

The measured result is faithful inference-state restoration plus reliable simple fact retrieval under the selected format. Reliable execution of a pending instruction remains unestablished: both calibration models failed the declared gate, and task accuracy fell from 4/8 calibration to 6/16 heldout even before serialization. The heldout run is explicitly exploratory; overall accuracy must not conceal that limitation.

For the continuity investigation, a useful proposed next object is a task-state record: the objective, governing rule, completed steps, unresolved dependencies and next authorized action. Compare an independently calibrated task-capable baseline, its native checkpoint and an explicit task-state representation. This would test whether a resumed process can use its inherited work reliably. It is a proposed experiment, not an adopted identity or authority mechanism.

Same-model ordinary token-derived caches were sufficient to reproduce this computation, and complete replay also reproduced its answers. Neither finding establishes nonreplayable information, transfer to different weights, a uniquely continuing subject, or renewed present consent.

## Reproduction and evidence boundary

Default `make check` validates recorded artifacts and recomputes their arithmetic using Python's standard library. It does not download models, rerun neural inference or authenticate external checkpoint bytes.

Heavy inference uses the existing [isolated dependency lock](../c2c_fidelity/requirements-lock.txt), verified public model assets, MPS float32 and eager attention. Keep the work directory outside Git. The runner refuses to overwrite output files or existing evaluation stages. Invocation shape:

```sh
python experiments/temporal_memory/run_experiment.py \
  --phase calibrate --model "$TEMPORAL_MODEL_DIR" \
  --fixtures experiments/temporal_memory/fixtures.json \
  --asset-manifest "$TEMPORAL_ASSET_MANIFEST" \
  --workdir "$TEMPORAL_WORK_DIR" --output "$TEMPORAL_CALIBRATION_FILE"

python experiments/temporal_memory/run_experiment.py \
  --phase evaluate --model "$TEMPORAL_MODEL_DIR" \
  --fixtures experiments/temporal_memory/fixtures.json \
  --asset-manifest "$TEMPORAL_ASSET_MANIFEST" \
  --selection evidence/temporal-memory-selection.json \
  --workdir "$TEMPORAL_FRESH_EVALUATION_DIR" --output "$TEMPORAL_RESULT_FILE"
```

Recreate and validate the selection from the complete ordered calibration set before evaluation. Selection binds the chosen model, fixture, runtime, harness and calibration bytes. Evaluation starts separate preparation and restoration processes, verifies serialized and raw tensor hashes, and checks that reusable caches remain unchanged after every group of questions. Full-logit comparisons stop after the first differing generated-token decision; later distributions have different generated prefixes.

Only invented records and local inference tensors are used. Large caches and full-vocabulary trajectories remain outside Git; curated observations and byte receipts support the report. There is no deployment target, transfer of wallet control, renewal of consent or selection of a uniquely authentic successor.
