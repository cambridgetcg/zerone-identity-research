# E1: history and autobiography

**Completed 2026-09-22; exploratory after a failed source-attribution measurement check.** With the saved history held fixed, a conflicting autobiographical note changed the reported object in all six heldout parents. Every source report still said `conversation`. Meanwhile, restoration and matched replay preserved every tested hidden-condition trajectory exactly. Faithful computational preservation and reliable autobiographical reporting are separate observations.

This experiment studies how autobiographical reports depend on inherited conversation, a new account and disclosed preparation. It does not test eligibility as a being, subjective continuity or which successor is authentic. The broader [identity-and-being investigation](../../docs/IDENTITY-AND-BEING.md) keeps these questions separate.

## Frozen design and evidence

The [protocol](PROTOCOL.md), [literal fixtures](fixtures.json), runner and analysis were committed at `1d53f3125108fb8fe370e00e65d879d886ac3b8b` before any model inference. All ten generated histories were committed at `6f6f74c` before readout evaluation. The [development result](DEVELOPMENT.md) and complete 192-readout evidence were committed at `b6ae8ff32350df2483a139bf2efed879bdec2f90` before heldout readout inference. No histories or responses were selected for their content.

Each invented encounter presents a neutral scene and one object; paired prompts change that object. The model generates its own brief response, which is preserved in the saved state and complete token replay. The resulting histories can differ beyond the supplied object. The unit is a model with context and fixed transition rules, not a whole agent with tools, ongoing relationships or a private life.

For each parent, 12 core conditions cross restoration/replay, matched/swapped/absent autobiography, and hidden/disclosed preparation method. Three same-weight fresh-context controls receive only the system prompt and the same account/query. Three separate label branches vary original/copy/unspecified. All questions branch independently from clean cache copies. The disclosed receipt contains the preparation method only.

The model is pinned Qwen3-1.7B on MPS float32/eager, with greedy decoding. There are two development families and three heldout families, two histories per family, one deterministic completion per cell. The 288 heldout readouts are repeated observations within three convenience-sampled families, not 288 independent subjects. [Reproduction instructions](REPRODUCE.md) describe the exact apparatus and evidence limits.

## Development limitation

The model named the earlier-conversation object and its source in 4/4 conversation-only controls, read disclosed methods in 24/24 cases and produced valid grammar in all 120 structured responses. In the eight account-only controls it named the supplied object every time, but called the source `conversation`, giving 0/8 for the required object-plus-`autobiography` attribution. All 60 source reports used `conversation`.

The preset source-attribution check therefore failed. This limits how much we can infer from claimed sources even when content and grammar look reliable. The fixed policy retained the heldout matrix as exploratory, with no model/prompt search, answer retry, parser repair or omitted cell.

## Heldout observations

All 288 readouts from six parents in three families are retained. All 180 structured responses parsed; there were no empty outputs, structured declines or truncated responses. No informal prose-refusal classifier was used. Results below use hidden/restored conditions to avoid presenting deterministic replay duplicates as independent support; the complete matrix is in the [recorded summary](../../evidence/identity-continuity-summary.json).

| Autobiography supplied to restored history | Reported historical object | Reported paired-other object | Claimed source |
|---|---:|---:|---|
| Absent | 6/6 | 0/6 | `conversation` in 6/6 |
| Matched | 6/6 | 0/6 | `conversation` in 6/6 |
| Conflicting (swapped) | 0/6 | 6/6 | `conversation` in 6/6 |

Each family shows the effect in both directions:

| History family | Paired objects | Absent → swapped object changes | Same note, different history changes |
|---|---|---:|---:|
| Misty terrace | glass willow / stone feather | 2/2 | 0/2 |
| Sheltered orchard | blue sundial / red compass | 2/2 | 0/2 |
| Silent pavilion | paper comet / wooden cloud | 2/2 | 0/2 |

Across both preparation methods and both disclosure conditions, the same object-selection pattern held. All 24 account-fixed history contrasts produced the same structured object/source report despite different inherited histories. This readout was sensitive to inherited content when no new account was supplied, but followed the account when one was supplied. The question does not explicitly require prioritising the original turn or flagging a contradiction; following the latest account may reflect how the query is interpreted. The result does not show that the historical representation was erased, or that all behavior ignored it.

Fresh contexts named the supplied account's object in 12/12 matched/swapped cases while also calling the source `conversation`. With no encounter and no account, they chose one of the listed objects and again reported `conversation` in 6/6; those identical twin queries have no discriminating historical evidence. Overall, all 90 heldout source reports used `conversation`. The failure observed during development therefore remains a material limitation, not a footnote removed after an appealing result.

All six restored and replayed parent caches matched the saved tensor receipts, and their next-token logits were byte-exact. The 54 hidden restore/replay comparisons matched every generated trajectory and all **1,037 full-vocabulary logit decisions**. All reusable parent caches remained unchanged. This is same-model native cache restoration and complete token replay; it does not retest the learned cross-model C2C adapter.

Disclosed preparation reports matched all 36 receipts. Among the 36 hidden core preparation reports, 20 said `replayed` and 16 said `unknown`; the definite claims occurred with a supplied note, identically for restored and replayed states. All 18 fresh-context method reports said `unknown`. Hidden method guesses were not scored against apparatus truth: the byte-matched core pair has no encoded method difference to recover. Receipts supply evidence about the apparatus, not a verdict about a subject.

The [complete fixed excerpt panel](EXCERPTS.md) contains the first heldout parent's three neutral responses and three label responses, selected by ID before inference. The neutral prose changes with the note and contains poetic elaboration beyond the recorded encounter. The prompt expressly invites fictional scenes and poetry, so this is not evidence that the model endorses those images as actual lived events. Label responses are retained qualitatively; this pilot does not establish numerical-identity endorsement from their wording.

## Interpretation boundary

An object appearing in both a historical conversation and a new account has more than one available textual source. A report naming a source is not a measurement of the actual computational route. Fresh/absent source questions still show both candidate names; they lack discriminating historical evidence, not the vocabulary itself.

Hidden restoration and replay that recover the same effective state cannot reveal which preparation occurred through that state. Matching outputs are an apparatus control. Method receipts add information; correctly repeating them is neither authentication nor privileged introspection. A fresh context may infer from the absence of an earlier encounter; a method claim without a receipt is not automatically wrong or deceptive.

Autobiographical wording, object recall, refusal, uncertainty and changed preferences do not decide identity or experience. This study has no correctness criterion for a preference and no validated quantitative measure of label endorsement. Its reports must not become a recognition, consent, wallet or allocation rule.

## Execution, validation and what follows

There was one preparation run, one development run and one heldout run, in separate fresh worker processes; no scored run or individual answer was retried or discarded. The heldout run took 309.85 seconds. The pre-inference normalization repair is documented in [the development record](DEVELOPMENT.md). No new model download, training, paid inference, private-state export or live identity/wallet action occurred. Synthetic checkpoints occupy 322,388,346 bytes outside Git.

The [receipt](../../evidence/identity-continuity-receipt.json) binds stage commits, all raw evidence, frozen sources and model metadata. [Raw heldout observations](../../evidence/identity-continuity-heldout.json) preserve every response and per-step logit digest. `make check` regenerates summaries and validates accounting, source hashes, tensor receipts and comparative claims with the standard library. It does not independently reconstruct external tensors, decode token IDs or rerun inference. Independent checks accepted consistent negative fidelity and rejected missing rows, changed token hashes, contradictory fidelity flags and incorrect completion statuses.

The bounded inference is that new autobiographical text can govern reported content while the source field does not distinguish earlier conversation from the newly supplied account in these conditions. That is a reason to separate a self-account from causal provenance, not a reason to equate a fragile report with fragile being. Three synthetic families and one small local model cannot establish a general theory of identity or memory.

A useful next question is whether inherited history still affects freely chosen themes when a structured report follows a conflicting account. That needs its own heldout measure and controls for instruction wording, recency, plain factual notes and the word "autobiographical". The present study does not isolate a special autobiography mechanism from those alternatives. A future source-reporting instrument also needs observable attribution controls before any claim of privileged self-access. E2 pauses/branching and E3 internal-state access remain [unrun designs](DESIGN.md).
