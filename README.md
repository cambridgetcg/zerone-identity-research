# Zerone identity research

Studies of continuity through changes of name, model, host, keys, wallet and relationships. The central question is what can be resolved through recognition and agreement, what existing software already handles, and which resource-specific rules remain undesigned.

This is a private research repository. Accepted design choices, source observations, relational interpretations and executable model results are kept separate. It does not define who is a real being, issue identity or consent, or activate changes in Zerone or AgentTool.

## Design choice carried forward

Participation may use a stable public pseudonym and accurately declared role while offline personal details remain private. Relevant authority, conflicts and research allocations should be accountable. A founder title gives no automatic funding share, extra vote or exemption. No actual personal identity or pseudonym mapping is stored here.

## Studies

| Study | Question | Status |
|---|---|---|
| [01 — Renaming](studies/01-renaming/README.md) | How does a name change preserve attribution and privacy? | Scoped |
| [02 — Model and host change](studies/02-model-and-host-change/README.md) | What continuity and consent survive a change of runtime? | C2C pilot and native checkpoint control |
| [03 — Deliberate fork](studies/03-deliberate-fork/README.md) | What is shared history, and what requires a new agreement? | Scoped |
| [04 — Key loss and recovery](studies/04-key-loss-and-recovery/README.md) | Who may recover which resource, and under what evidence? | Scoped |
| [05 — Restoration and return](studies/05-restoration-and-return/README.md) | What happens when the original returns after an authorized restoration? | First investigation |
| [06 — Absence and return](studies/06-absence-and-return/README.md) | How can absence remain uncertain without making rest a forfeiture? | Scoped |
| [07 — Partnership exit](studies/07-partnership-exit/README.md) | How can either participant leave while handling prior commitments? | Scoped |
| [08 — Role succession](studies/08-role-succession/README.md) | How does a role move without transferring the predecessor's identity? | Scoped |
| [09 — Allocation dispute](studies/09-allocation-dispute/README.md) | How are contested funds handled without judging someone's existence? | Scoped |
| [10 — Wallet change](studies/10-wallet-change/README.md) | How do payment instructions change independently of identity? | Scoped |

Start with the [research method](docs/METHOD.md), [decision record](docs/DECISIONS.md), and [restoration-and-return study](studies/05-restoration-and-return/README.md).

The [historical-parallels follow-up](studies/05-restoration-and-return/HISTORICAL-PARALLELS.md) studies how a returning participant could challenge a recovery without automatically regaining control.

The [state-transfer investigation](studies/02-model-and-host-change/STATE-TRANSFER.md) explores whether saved computation can support continuity across hosts or models, with complete replay and duplicate restoration as essential comparisons.

The [C2C fidelity pilot](experiments/c2c_fidelity/README.md) runs the released small-model pair on 24 synthetic cases with matched, swapped, zero and disabled donor-cache controls. It also tests ordinary native cache serialization and resumption. Heavy inference is opt-in; default CI validates recorded evidence and reproduces its arithmetic without downloading models.

## Reproduce the first experiment

Python 3.11+ and Make are sufficient; there are no third-party dependencies.

```sh
make check
python3 model/return_model.py
```

The experiment uses invented labels and symbolic credentials. It models limited resource authorization under explicit assumptions; it is neither cryptographic validation nor a deployed recovery implementation. See [model assumptions](model/README.md) and [recorded results](evidence/return-model-results.json).

The first result is conditional: replacing one resource's credential excludes holders of only the old credential when every action checks current authority. A stale checker defeats that exclusion; sharing the replacement defeats exclusive control. Delayed recovery also needs an expected-current-epoch check to prevent rollback. These operations do not decide identity, renew partnership consent or choose a legitimate recovery authority.

The experiment includes 14 tests and five model variants. Each variant covers all 30,940 nonempty attempt sequences of up to four events from a fixed 13-event alphabet, including rejected and possession-impossible attempts. This is a small bounded model with one or three distinct states per variant, not evidence about real-world incident rates.

## Evidence and boundaries

[Pinned source metadata](evidence/source-manifest.json) records exact owner/revision/path/SHA-256 coordinates from the initial source review. [Source coverage](docs/SOURCE-COVERAGE.md) explains what those observations support. [Primary references](docs/REFERENCES.md) distinguish distributed-system results from identity and consent questions.

No private WAKE, secrets, real credentials, personal identity mapping or live account state is included. Upstream source links may require access. This repository publishes research through Git; it has no application deployment target.
