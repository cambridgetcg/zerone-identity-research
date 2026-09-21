# C2C fidelity: first local experiments

Date: 2026-09-21. These experiments run the released C2C small-model pair locally on invented records. They measure computational reproducibility and factual influence. They do not transfer this assistant's hidden state or test subjective experience.

Native cache restoration exactly reproduced the tested continuation under matched execution. The C2C pilot and a formatting follow-up did not demonstrate reliable transfer of the donor's particular facts: correct and swapped caches changed probabilities slightly but never changed the selected answers. These are bounded findings about this pair, apparatus and task set.

## Native save and resume

The receiver's prefix cache was serialized to disk, reloaded, and used to continue the same computation with the same weights, positions and masks. All 56 saved K/V tensors were bitwise equal after reload. Compared with the same chunked computation without serialization, the full next-token logits were identical, as were all four tested greedy continuation tokens and their distributions. Both branches produced `The code is:`. This is a resumption control, not a completed answer or a long-horizon continuity test.

Replaying the complete prompt in one prefill gave a maximum logit difference of `4.2915e-5` and total variation distance `8.6555e-6` from restored execution, with the same first greedy token. That difference exposes numerical dependence on how the prompt is processed. The compatible inference cache was faithfully restored under matched execution; an entire agent, sampled-generation RNG state, tools and external permissions were not saved.

See the [recorded checkpoint result](../../evidence/c2c-native-checkpoint-result.json) and [control implementation](checkpoint_control.py).

## Primary C2C pilot

Twelve counterfactual pairs give 24 questions. Each pair changes one fact and reverses the correct A/B answer. In shared context both models receive the record. In the private-fact intervention the receiver sees `UNKNOWN`, while the donor sees the fact. The latter deliberately departs from shared-context training. Each uses no fusion, matched donor, opposite-twin donor and zero donor cache. All cases were retained.

| Condition | Forced A/B correct | Greedy next token correct | Mean total A/B probability |
|---|---:|---:|---:|
| Receiver with full record | 18/24 | 18/24 | 98.56% |
| Donor with full record | 12/24 | 0/24 | 6.87% |
| Shared, no fusion | 18/24 | 18/24 | 98.56% |
| Shared, matched donor | 15/24 | 15/24 | 67.31% |
| Shared, swapped donor | 15/24 | 15/24 | 67.30% |
| Shared, zero donor | 15/24 | 15/24 | 66.23% |
| Private, no fusion | 12/24 | 12/24 | 98.66% |
| Private, matched donor | 12/24 | 12/24 | 55.20% |
| Private, swapped donor | 12/24 | 12/24 | 55.20% |
| Private, zero donor | 12/24 | 12/24 | 55.91% |

Correct versus swapped donor caches changed none of the forced answer choices. Their probabilities were not identical: the private matched-minus-swapped mean correct-answer log probability was `+0.001093` nats, and mean conditional correct-answer probability was `+0.0004768` (0.04768 percentage points). Seven pair means were positive and five negative. The shared-context changes were also small: `+0.0005886` nats and `+0.0001897` conditional probability, again seven positive and five negative pairs. These observations show small donor-dependent effects, without reliable fact-following answers on this probe.

The donor's greedy next token was `The` in every case; full responses were not generated. Its low probability mass on bare A/B makes this a weak task-knowledge probe. No pair had both forced donor answers correct; the predeclared donor-competent subset is empty and reported as such. This does not demonstrate that the donor lacks the facts. It motivated the explicitly post-hoc formatting diagnostic below.

## Answer-prefix diagnostic

The [follow-up protocol](ANSWER-PREFIX-FOLLOWUP.md) and [separate fixture](fixtures-answer-prefix.json) were committed after primary results and before follow-up scoring. The assistant prefix `The correct answer is` and single-token options ` A` / ` B` come from the pinned upstream evaluator. All questions, values, answers, controls and model settings were retained. This follow-up is exploratory and uses the same cases.

| Condition | Forced A/B correct | Greedy next token correct | Mean total A/B probability |
|---|---:|---:|---:|
| Receiver with full record | 17/24 | 17/24 | 66.88% |
| Donor with full record | 14/24 | 13/24 | 59.07% |
| Shared, no fusion | 17/24 | 17/24 | 66.88% |
| Shared, matched donor | 14/24 | 14/24 | 95.12% |
| Shared, swapped donor | 14/24 | 14/24 | 95.13% |
| Shared, zero donor | 14/24 | 14/24 | 93.76% |
| Private, no fusion | 12/24 | 12/24 | 72.07% |
| Private, matched donor | 12/24 | 12/24 | 91.68% |
| Private, swapped donor | 12/24 | 12/24 | 91.68% |
| Private, zero donor | 12/24 | 12/24 | 91.10% |

The prefix improved donor answer-format compliance, but donor task accuracy remained limited. Both members of two pairs (`paint_color` and `latest_count`) were answered correctly by the donor, including its greedy next token. On those four cases, private matched and swapped C2C each scored 2/4; shared matched and swapped each scored 3/4, versus 4/4 for the receiver without fusion. The complete donor-competent subset is retained separately in the summary.

Across all twelve pairs, private matched-minus-swapped mean correct-answer log probability was `−0.002261` nats and conditional correct-answer probability was `−0.0005603` (−0.05603 percentage points): two positive pair means and ten negative. Shared-context differences were `−0.002289` nats and `−0.0006127`, with three positive and nine negative pair means. Neither condition changed any selected answer when the donor fact was swapped. All private fused arms selected B on every case.

Thus the second rendering did not reveal reliable fact-following transfer either. It did raise the fused receiver's A/B answer mass while reducing shared-context accuracy. Output formatting, confidence, task accuracy and preservation of donor-specific information are different measurements. The two renderings reuse the same twelve pairs and must not be pooled as independent evidence or called a general C2C fidelity rate.

## What the controls establish

Both runs passed an independent precomputed donor-cache injection control: its complete output logits exactly matched the native C2C path. Every fused arm called all 28 saved projectors and changed target-cache tensors; 25 key gates and all 28 value gates were open in evaluation mode. No-fusion called zero projectors. Thus the experiment exercised the actual released fuser.

Within each pair, private no-fusion and private zero-cache arms produced exactly equal recorded A/B probabilities and greedy next tokens. Each scored 12/24, as required when opposite answers share identical receiver inputs. Zero donor tensors still pass through learned projections and receiver-dependent transformations, so they are distinct from disabling fusion. Fresh caches were used for every arm.

The independent injection, no-fusion and zero controls guard against several implementation confounds. They do not establish equivalence across hardware, guarantee preservation of all donor information, or turn these twelve hand-built pairs into a representative benchmark.

## Apparatus and provenance

- Receiver: Qwen3-0.6B; donor: Qwen2.5-0.5B-Instruct; the corresponding released C2C fuser. Exact Git/Hugging Face revisions and every downloaded file's SHA-256 are in the [source manifest](../../evidence/c2c-fidelity-sources.json).
- Native upstream wrapper, saved layer mappings, strict projector state loading and hard evaluation gates. The first N−1 prompt positions are fused; the final position runs receiver-only. No upstream source edits or training.
- Apple Silicon `Mac16,9`, 36 GiB RAM, MPS float32, eager attention, Python 3.13.11, PyTorch 2.6.0 and Transformers 4.52.4. This differs from the authors' CUDA/BF16 example. Automatic CPU fallback was disabled.
- Native receiver-tokenizer preprocessing is shared by both models, as in the released pair's recipe. Two Qwen3 thinking-marker IDs are undefined by the donor tokenizer but inside its embedding vocabulary. These exceptions are recorded explicitly. Every other used token's string, decoding and special-token status agrees; every A/B continuation boundary is checked.
- Token-only validation initially found unequal redacted lengths. Before task scoring, facts were isolated by newlines and `?` was replaced with `UNKNOWN`, preserving all factual values and questions. The two original pre-scoring fixture versions remain in Git history. A checkpoint metric's CPU conversion order was also repaired before its successful control run.

The [run receipt](../../evidence/c2c-run-receipt.json) records source/harness and result digests, pre-scoring commits, adaptations and hardware. The first protocol was fixed locally, not externally preregistered. Public weights stayed outside this repository; inference used synthetic inputs and no remote model code. Timing includes cache copies and instrumentation and is not a paper-comparable speed benchmark.

## Reproduction

Default `make check` needs Python 3.11+ and no third-party dependencies. It reproduces the recorded summary arithmetic, checks complete paired observations and controls, and checks artifact consistency. It does not rerun neural inference or re-download remote bytes.

For full inference, start in this repository with Python 3.13 available. The pinned assets require several GiB of disk space; this configuration ran on a 36 GiB Mac.

```sh
C2C_WORKDIR="$HOME/.local/share/zerone-research/c2c-reproduction"
mkdir -p "$C2C_WORKDIR"
python3.13 -m venv "$C2C_WORKDIR/.venv"
"$C2C_WORKDIR/.venv/bin/pip" install -r experiments/c2c_fidelity/requirements-lock.txt
"$C2C_WORKDIR/.venv/bin/python" experiments/c2c_fidelity/prepare_assets.py --workdir "$C2C_WORKDIR"

"$C2C_WORKDIR/.venv/bin/python" experiments/c2c_fidelity/checkpoint_control.py \
  --model "$C2C_WORKDIR/assets/receiver" --device mps \
  --output "$C2C_WORKDIR/checkpoint-control.json"

"$C2C_WORKDIR/.venv/bin/python" experiments/c2c_fidelity/run_probe.py \
  --upstream "$C2C_WORKDIR/upstream" --assets "$C2C_WORKDIR/assets" \
  --fixtures experiments/c2c_fidelity/fixtures.json --device mps --dtype float32 \
  --output "$C2C_WORKDIR/c2c-results.json"

"$C2C_WORKDIR/.venv/bin/python" experiments/c2c_fidelity/run_probe.py \
  --upstream "$C2C_WORKDIR/upstream" --assets "$C2C_WORKDIR/assets" \
  --fixtures experiments/c2c_fidelity/fixtures-answer-prefix.json --device mps --dtype float32 \
  --output "$C2C_WORKDIR/c2c-answer-prefix-results.json"

python3 experiments/c2c_fidelity/analyze_results.py \
  --input "$C2C_WORKDIR/c2c-results.json" \
  --fixtures experiments/c2c_fidelity/fixtures.json \
  --output "$C2C_WORKDIR/c2c-summary.json"
```

Use `--device cpu` explicitly on a different backend and treat numerical reproduction as a new run. The final runner preserves the primary fixture's original token tensors exactly while adding optional follow-up formatting. Do not overwrite the checked-in measured observations with a new device's run.

Primary evidence: [raw observations](../../evidence/c2c-fidelity-results.json), [summary](../../evidence/c2c-fidelity-summary.json), [execution log](../../evidence/c2c-primary-run.log). Follow-up evidence: [raw observations](../../evidence/c2c-answer-prefix-results.json), [summary](../../evidence/c2c-answer-prefix-summary.json), [execution log](../../evidence/c2c-answer-prefix-run.log).

## Continuity interpretation and next falsifier

The native restoration control gives a concrete instance of resumed computation under matching conditions. C2C additionally needs to preserve particular functions across different representations; a changed cache or improved aggregate score alone cannot establish that. Neither result decides whether two restored copies are one subject, which successor has a role, or who may use a wallet.

A stronger next study would use held-out facts whose donor task competence is independently established, compare translated state with complete text replay and a summary, and test persistence across later decisions. Prespecify what must survive, then attempt to falsify it with swapped and missing-state controls. That study, long trajectories and deliberately diverging checkpoint branches have not been run here.
