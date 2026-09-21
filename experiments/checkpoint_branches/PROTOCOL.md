# Native checkpoint branches: protocol fixed before scoring

Date: 2026-09-21. Follow-up to the first C2C fidelity pilot. This is a locally fixed exploratory protocol, not an external preregistration. It studies inference-state restoration and externally supplied branch updates, with no identity or succession policy adopted.

## Questions

1. Does loading the same saved prefix cache in two fresh processes reproduce the original computation when subsequent inputs are identical?
2. After different later updates, can both branches answer questions about their common earlier record and their different current records?
3. Do complete replay, a different parent's cache and redacted history explain the result?

Two valid restorations from the same checkpoint need not identify a uniquely authentic successor. That is an interpretation of the experiment's structure, not a test of subjective identity. Different branch inputs are supplied by the experimenter; divergence is not spontaneous agency.

## Fixed apparatus and cases

Reuse the pinned Qwen3-0.6B receiver and its verified public files from [the existing source manifest](../../evidence/c2c-fidelity-sources.json). Use the existing isolated Python 3.13.11 / PyTorch 2.6.0 / Transformers 4.52.4 environment, MPS float32, eager attention, evaluation mode, six CPU threads and greedy decoding. No C2C fuser, new weights or training is used in this experiment. CPU execution requires a separately recorded explicit run, with no automatic fallback.

[Fixtures](fixtures.json) contain eight parent records in four counterfactual twin pairs. Each records one mutable fact and one persistent fact. A twin changes both old values, preserving record structure. Each parent yields two branches whose supplied updates assign different new values to the mutable fact. Each branch is asked three independent A/B questions: original mutable value, persistent value and current mutable value. There are 48 branch questions, balanced 24 A / 24 B. Redacted histories replace both old values with `UNKNOWN` and are identical within twins.

Verify all parent/twin/redacted token lengths, full rendered token streams at each cache boundary, and the single-token A/B continuation before scoring. Do not silently pad, drop cases or tune questions after observing outcomes. Expected answers are scoring metadata only. A tokenization repair before scoring must be recorded.

## Execution and controls

A preparation process computes each parent prefix, writes its K/V tensors plus compatible boundary/runtime metadata, and retains an untouched in-memory reference. Save only synthetic state. Record hashes for serialized files, raw tensors, model assets, inputs and executed source. Binary caches and full-logit reference tensors remain outside Git.

From each original in-memory cache, ask its original and persistent questions without an update (16 parent competence observations). Also obtain both branches' six reference answers using separate cache copies (48 uninterrupted branch reference observations).

For a longer deterministic control, append the fixed request `Describe the original record in a short paragraph of about 50 words.` to the parent. Greedily generate at most 32 tokens, stopping at the model's end token and recording the actual length. Save every tested step's full-vocabulary logits outside Git. This checks execution fidelity, not the factual quality of that generated paragraph.

Launch two new, independent local Python worker processes, one for each branch, sequentially to avoid resource contention. Each reloads the pinned weights and the same parent checkpoint bytes. It first repeats the identical paragraph continuation, then receives its branch-specific update. Record distinct process IDs, matching parent hashes and branch-cache hashes after their different supplied updates. Every question and comparison starts with a fresh cache copy. Verify parent data remains unchanged.

Each branch question has five arms:

| Arm | Computation | Purpose |
|---|---|---|
| Restored | Saved parent cache → own update → query | Tested restoration and retention |
| Same-chunk replay | Recompute parent → own update → query | Complete replay with matched execution boundaries |
| Full replay | Parent, own update and query in one prefill | Complete-information comparison with different numerical execution |
| Swapped parent | Counterfactual twin's saved cache → own update → query | Whether answers depend on the supplied earlier history |
| Redacted parent | Recompute unknown old values → own update → query | Whether old values can be recovered without their history |

The five arms give 240 scored branch observations. Keep both the intended parent's gold answer and the swapped cache's supplied-history answer. A successful swap control should follow its supplied old facts while preserving the branch's newly supplied value; lower accuracy against the intended original is not itself a reasoning failure.

## Measurements, criteria and falsifiers

- Compare tensor hashes, full-logit distributions and generated tokens for the identical-input control. Exact equality is required for an exact-resumption claim under this fixed apparatus; preserve and investigate differences rather than changing the tolerance after outcomes.
- Compare restored branch logits with the original in-memory reference and same-chunk replay. Full replay has no preassumed bitwise guarantee: report maximum absolute logit error, total variation distance and greedy agreement.
- For A/B probes, report forced-choice accuracy, unrestricted greedy next-token accuracy, conditional and absolute answer probabilities, total A/B probability mass, and results by original/persistent/current question type. Report parent competence and all failures; do not select only convenient cases.
- Redacted histories must give identical recorded answer distributions within twins for each branch and old-fact question, with opposite gold answers. Those 32 original/persistent questions therefore score exactly 16/32 under deterministic forced choice. Current-value questions are not subject to that chance constraint, because their information appears in the updates.
- Count branches answering all three questions, and parents whose two branches both answer their different current values correctly. Compare with swapped-parent supplied-history scores and complete replay. These are eight parents in four paired families, not 240 independent trials.

If identical restored inputs fail to reproduce the reference, investigate missing state, aliasing or numerical execution before claiming preservation. If old-fact answers do not depend on the earlier cache, the experiment has not demonstrated retained history. If branches fail to follow their new inputs, distinct cache hashes alone do not demonstrate functional divergence. If complete replay reproduces the behavior, reject any claim that this ordinary cache contains information unavailable in the complete inputs.

The experiment does not save an entire agent, transfer to different weights, test stochastic sampling, simulate an independent human controller, renew consent or authorize resources. Cross-model learning, text-summary comparisons, sampled RNG restoration, long-term interaction and broader identity questions remain separate follow-ups.
