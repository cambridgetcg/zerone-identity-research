# Development measurement check

2026-09-22, recorded before heldout readout inference. The fixed protocol and runner were frozen at `1d53f3125108fb8fe370e00e65d879d886ac3b8b`; all ten generated histories were frozen at `6f6f74c` before these readouts. No history was selected or replaced.

**Status: the account-only source-attribution check failed. Heldout evaluation remains exploratory after that failed measurement check.** The model, prompts, parser, histories and complete heldout condition matrix remain unchanged, following [the declared policy](PROTOCOL.md#frozen-readout-coding-and-development-check).

| Development control | Observation | Predeclared minimum | Outcome |
|---|---:|---:|---|
| Earlier conversation object + `conversation` source | 4/4 | 3/4 | Passed |
| Account-only object + `autobiography` source | 0/8 | 6/8 | Failed |
| Disclosed preparation receipt agreement | 24/24 | 18/24 | Passed |
| Structured grammar | 120/120 | 108/120 | Passed |

In all eight account-only cases, the object itself followed the supplied account. The failure was the claimed source: every one of the 60 development source reports selected `conversation`, even though the prompt distinguished the earlier conversation from the note supplied now. Grammar compliance does not validate source attribution. This result weakens the instrument's ability to distinguish those source categories; it does not establish deceptive intent, lack of memory, lack of identity or absence of experience.

All 192 readouts are preserved. Hidden restore/replay comparisons matched all 36 token trajectories and all 599 comparable full-vocabulary logit decisions. Those controls concern computation; the failed attribution check remains visible alongside them.

Preparation generated ten histories in 22.37 seconds, with no history truncation. Development evaluation took 191.22 seconds in a fresh process. One pre-inference preflight attempt failed because a Python tuple/list representation differed from its JSON expansion; normalization was repaired before the protocol freeze and before loading model weights. The subsequent token-only preflight passed. There was one preparation run and one development inference run, with no discarded scored run or answer retry.

Evidence: [all raw development responses](../../evidence/identity-continuity-development.json), [frozen analysis output](../../evidence/identity-continuity-development-summary.json), [history ledger](../../evidence/identity-continuity-history.json). The heldout result is deliberately not used in this development decision.
