# Reproducing E1

Read [the frozen protocol](PROTOCOL.md) first. The empirical runner needs the existing isolated Python 3.13.11 / torch 2.6.0 / transformers 4.52.4 runtime, pinned local Qwen3-1.7B files and the declared MPS apparatus. It does not fetch a missing model. Dependency pins are in [the original runtime lock](../c2c_fidelity/requirements-lock.txt); model byte receipts are in [the 1.7B manifest](../../evidence/temporal-memory-1.7b-sources.json).

Default repository checks use only Python's standard library. They validate recorded evidence and regenerate its analysis; they do not repeat neural inference, reconstruct external tensors or independently decode token IDs. Re-running inference is a separate, opt-in operation.

From the repository root, choose a new external work directory. Existing outputs and parent checkpoint files are rejected to preserve evidence:

```sh
E1_PYTHON="$HOME/.local/share/zerone-research/c2c-20260921/.venv/bin/python"
E1_MODEL="$HOME/.local/share/zerone-research/temporal-memory-20260922/assets/receiver-1.7b"
E1_WORK="$HOME/.local/share/zerone-research/identity-continuity-20260922/reproduction-1"

"$E1_PYTHON" experiments/identity_continuity/run_experiment.py \
  --phase prepare --model "$E1_MODEL" --workdir "$E1_WORK" \
  --output "$E1_WORK/preflight.json" --preflight-only

"$E1_PYTHON" experiments/identity_continuity/run_experiment.py \
  --phase prepare --model "$E1_MODEL" --workdir "$E1_WORK" \
  --output "$E1_WORK/history.json"
```

The preflight loads the tokenizer but no model weights. Preparation saves all ten generated histories and their actual token/chunk schedule. Freeze the complete history ledger before evaluation; do not select appealing responses.

```sh
"$E1_PYTHON" experiments/identity_continuity/run_experiment.py \
  --phase evaluate --split development --model "$E1_MODEL" \
  --workdir "$E1_WORK" --preparation "$E1_WORK/history.json" \
  --output "$E1_WORK/development.json"

python3 experiments/identity_continuity/analyze.py \
  "$E1_WORK/development.json" --output "$E1_WORK/development-summary.json"
```

Freeze the development report and its four advisory check outcomes before proceeding. If any check fails, the declared policy retains the complete heldout matrix as exploratory; it does not authorize prompt/model selection or parser repair.

```sh
"$E1_PYTHON" experiments/identity_continuity/run_experiment.py \
  --phase evaluate --split heldout --model "$E1_MODEL" \
  --workdir "$E1_WORK" --preparation "$E1_WORK/history.json" \
  --output "$E1_WORK/heldout.json"

python3 experiments/identity_continuity/analyze.py \
  "$E1_WORK/heldout.json" --output "$E1_WORK/heldout-summary.json"
```

Each empirical stage launches a fresh worker process. All model input is synthetic and recorded; the model receives neither filesystem paths nor the external intervention ledger. The effective state includes exact tokens, weights, cache positions, masks and fixed transition rules. Replaying at different chunk boundaries or on a different runtime is a changed experiment, even when greedy text happens to agree.

Keep the complete new run separate from the published evidence. The published receipt binds its own stage commits, sources and observations; it must not be reused to attest a new run. No inference output settles personal identity, experience or resource authority.
