# C2C fidelity pilot: protocol fixed before scoring

Date: 2026-09-21. The owner requested an actual experiment with C2C. This is a small synthetic pilot, not a reproduction of the paper's benchmarks or an identity experiment. Protocol and fixtures are fixed locally before the scored run; this is not an external preregistration.

## Apparatus

- Official C2C source: `thu-nics/C2C` at `9fa85d35a20e20469af3e282926491222f4f11e0`.
- Receiver: `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.
- Donor: `Qwen/Qwen2.5-0.5B-Instruct` at `7ae557604adf67be50417f59c2c2f167def9a775`.
- Released fuser: `nics-efc/C2C_Fuser` at `f01fc3258b305e280e04c7238f4f2cf31b7dc70d`, only `qwen3_0.6b+qwen2.5_0.5b_Fuser/final`.
- Upstream package pins: PyTorch 2.6.0 and Transformers 4.52.4. The run records transitive dependencies and device details.
- Intended execution: local Apple Silicon MPS, float32 and evaluation mode, with CPU float32 as an explicit fallback if the backend cannot run the native path. Any fallback or repair is recorded before interpreting scores. This differs from the authors' CUDA/BF16 example.

Load safetensors without remote model code. Load projector state dictionaries with `weights_only=True` and strict parameter matching. Use the native wrapper, saved mappings and hard evaluation gates. Fuse the first N−1 prompt tokens; process the final token without projection. Use the receiver tokenizer for both models, following this pair's released recipe (`is_do_alignment: false`); validate relevant token semantics before scoring. No fine-tuning or test-set adaptation.

## Cases

[Fixtures](fixtures.json) contain twelve counterfactual twin pairs, two cases each: retrieval, indirection, latest update, arithmetic, rules and associations. Each pair changes one fact and reverses the correct A/B answer. There are twelve A and twelve B answers. Expected-answer metadata is never rendered into a model input.

Apply the receiver's chat template with thinking disabled. Verify that A/B are distinct single-token continuations at the rendered boundary. Assert equal token lengths for both twins and the version replacing the fact with `?`; do not silently pad, change rotary positions or drop incompatible cases. A validation failure may be repaired before scoring, with the change recorded.

## Conditions

| Condition | Receiver input | Donor input/cache | Interpretation |
|---|---|---|---|
| Receiver reference | Full factual record | None | Receiver task competence |
| Donor reference | N/A | Full factual record | Donor task competence under the chosen rendering |
| Shared no fusion | Full factual record | Fusion disabled, same N−1/1 processing | Chunking/numerical control |
| Shared matched | Full factual record | Same factual record | Native shared-context use |
| Shared swapped | Full factual record | Opposite twin | Contradictory-cache intervention |
| Shared zero | Full factual record | Zero donor tensors | Nonsemantic fusion effects |
| Private no fusion | Fact replaced with `?` | Fusion disabled | Missing-information control |
| Private matched | Identical redacted input within each pair | Correct factual record | Exploratory transfer beyond the shared-context setup |
| Private swapped | Same redacted input | Opposite twin | Whether output tracks the specific donor fact |
| Private zero | Same redacted input | Zero donor tensors | Whether generic cache transformation drives the result |

Zero donor cache is not no fusion: the projector also uses receiver state and biases. Use fresh caches for every arm. Verify projection calls and nonzero target-cache changes. An independent precomputed matched-cache injection should match native computation before interpreting cache interventions.

## Measurements and decision criteria

Record raw per-case A/B conditional probabilities, their full-vocabulary probability mass, forced A/B choice, actual greedy next token, ground-truth probability/log probability and elapsed execution time. These are forced-choice probes, not unconstrained answer accuracy. Device synchronization and instrumentation are part of measured latency, so report no paper-comparable speedup.

Compare matched versus swapped correct-answer log probabilities and the direction of counterfactual changes. Count pairs for which both twins are answered correctly. Report all cases plus a separately labelled subset where the donor gets both twins right; retain all donor failures.

The private receiver input is identical within each pair, while gold answers are opposite. Deterministic no-fusion forced-choice accuracy must be 12/24; zero-cache outputs should likewise match within pairs. A violation requires investigation for leakage or nondeterminism.

Native checkpoint control: for one synthetic input, compare the same receiver's ordinary continuation with prefix-cache serialization/reload and processing the remaining input. Record tensor/logit differences, probability-distribution differences and several greedy tokens. This preserves only the tested inference state, not an entire agent or a subject. Record any numerical tolerance before interpreting differences.

Improved shared-context scores show task behavior, not preservation of donor-specific state. Private matched caches improving over counterfactual caches and following the changed fact would support bounded information transfer. Private failure would not refute the paper's shared-context results. Twelve synthetic pairs cannot establish a universal fidelity rate. Neither matching outputs nor cache equality settles subjective continuity, independent control, current consent or allocation rights.
