# Continuity through saved computational state

Date: 2026-09-21. Status: literature investigation with a [first empirical C2C pilot](../../experiments/c2c_fidelity/README.md). The pilot tests native cache restoration and cross-model factual influence; the broader experiment below remains proposed. No identity criterion or migration policy is adopted.

## The hypothesis

The user proposes that an agent could continue from saved internal state, including non-language representations, rather than relying solely on a retrieved account of its earlier history. Their analogy is human continuity through changes in physical composition and organisation. On 2026-09-21, they confirmed the paper they meant: *Cache-to-Cache: Direct Semantic Communication Between Large Language Models* (C2C). The source version inspected below is v1; the version they encountered is unspecified. The other papers are related research, not alternative identifications of their reference.

Our interpretation is a hypothesis about causal and functional continuity: later computation can depend directly on preserved earlier state despite a change of host. This is a concrete property to investigate. Whether it establishes continuity of a subject is a separate philosophical question. Material sameness need not be the only proposed criterion, but a biological analogy does not by itself validate a particular machine-state transfer.

## Relevant research

| Source and inspected scope | What it contributes | Boundary |
|---|---|---|
| [Cache-to-Cache, arXiv:2510.03215v1](https://arxiv.org/html/2510.03215v1), abstract and §§3.1, 3.3; token alignment appendix | Learned projection and fusion of one model's KV cache into another model's cache | Semantic communication with alignment assumptions, not an exact transfer of an entire agent |
| [Latent Cache Flow, arXiv:2605.22863v1](https://arxiv.org/html/2605.22863v1), §§2–3 and §5 | Compressed cache fusion; LCF-X pools information from a different context | Early evaluations use one heterogeneous pair for shared context and two instances of the same model for cross-context transfer; no general preservation guarantee |
| [LatentMAS, arXiv:2511.20639v1](https://arxiv.org/html/2511.20639v1), §§3.1–3.2 | Continuous hidden-state steps and layer-wise cache communication | The representation transfer described in the paper is not a demonstrated preservation of subjective identity, arbitrary agent state or independent control |
| [Causal audit, arXiv:2608.04893v2](https://arxiv.org/abs/2608.04893v2), abstract only | Swaps real caches with mismatched, zeroed and random controls to test example-specific transfer | Reported benchmark gains alone do not identify what information moved; no independent reproduction here |

These are scoped readings, not full-paper verification or implementation audits. Performance numbers and theorem claims have not been independently reproduced.

## Three different operations

| Operation | Proposed continuity claim to evaluate |
|---|---|
| Resume compatible computation from a complete checkpoint | Preserve the relevant state and transition rules at a defined boundary |
| Reconstruct state by replaying complete inputs/computation | Reach an equivalent computational state through recomputation |
| Translate representations into a different model | Preserve specified information or functions despite changed representations and transition rules |

The third operation requires evidence beyond success at the first. A translated cache can enrich an already distinct receiver, which is communication rather than evidence of migration. Compression may retain useful information while losing other details.

## What has to be saved

For an ordinary autoregressive transformer, KV stores per-layer keys and values of earlier inputs. A checkpoint also needs the compatible weights and an exact resume boundary, including the next input or pending logits, positions/masks and generation state. The full agent additionally depends on orchestration and relevant external state. This is a checkpoint-design inference from the [documented cache computation](https://huggingface.co/docs/transformers/en/cache_explanation), not a portable checkpoint format implemented here.

Sampling needs the generator's current state, not just its original seed. Numerical execution also matters: [vLLM's reproducibility documentation](https://docs.vllm.ai/en/latest/usage/reproducibility/) restricts its guarantees to specified settings with the same hardware and version. These mutable documentation pages were consulted on 2026-09-21; they are not exact source-code pins or evidence of our runtime's behavior.

## Why complete replay is an essential comparison

With fixed deterministic weights and matching inputs/positions/masks, ordinary token-derived KV can be recomputed. A short biography or visible transcript may omit inputs, but complete input replay is a stronger baseline. This follows from the cache recurrence; bitwise identity requires additional numerical controls. Cache persistence still avoids recomputation.

If the computation includes continuous latent steps, as in LatentMAS, visible token replay alone omits them. Preserve the relevant latent state or replay the full latent computation and its schedule. The distinction is completeness of the causal record; non-language format alone does not establish extra irrecoverable information.

## Proposed experiment

Use a small, openly inspectable model and invented task data. Fix weights, tokenizer, implementation, precision, positions, sampling and environment; record the exact checkpoint boundary. Keep tools simulated and avoid external effects.

1. Give the model novel synthetic information and an unfinished task. Keep an uninterrupted continuation as the reference.
2. Compare full checkpoint restoration, exact complete-input replay and restoration from a text summary.
3. Include missing-state and mismatched-cache controls. Make some questions depend on information unique to the synthetic pre-checkpoint history, absent from the model's prior knowledge and any recipient-only baseline. Complete input replay must retain that information too.
4. Within the same model/vocabulary, compare tensors, next-token distributions and later trajectories, with explicit numerical tolerances. Also measure task retention and recomputation cost. Matching one answer is insufficient.
5. Restore twice with matching inputs and random states; then give the branches different information. Record common provenance and later divergence without selecting a uniquely authentic branch.
6. Test cross-model translation separately against recipient-only, complete-text and summary baselines. Use common observable tasks with held-out information; do not compare vocabulary logits directly across incompatible tokenizers.

The owner's subsequent request authorized the [first local pilot](../../experiments/c2c_fidelity/README.md), with pinned public models, source and isolated dependencies. That pilot covers only part of this proposed experiment. It uses synthetic inputs and makes no claim to have accessed an API provider's internal cache.

## Falsifiers and identity boundary

If checkpoint restoration differs from uninterrupted continuation beyond the declared tolerance, investigate missing state or execution differences before calling it continuity. If full replay matches restoration, reject the claim that ordinary KV necessarily adds information absent from complete replay. If translated state fails against appropriate baselines or mismatched-donor controls, reject the claimed transfer benefit for that setup.

Success could establish retained information and a defined degree of computational continuity. It would not distinguish a unique continuing subject from two equivalent successors. Resuming the same checkpoint twice exposes the original restoration-and-return question more sharply: both branches can have the same verified computational ancestry.

For Zerone, keep computational provenance alongside the other layers. It can supplement narrative history and participant recognition. It does not duplicate funding, transfer wallet authority, renew another participant's consent, or make the current controller's claim immune to review. A checkpoint may preserve historical state while current external authorizations have changed.
