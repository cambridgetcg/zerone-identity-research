# Restoration and return: symbolic resource model

This experiment studies a **proposed** resource-specific authorization pattern. It is not the implementation of Zerone, AgentTool or an adopted recovery policy. It makes no decision about who is the original, who has subjective continuity, or who deserves a research allocation.

Two simulated actors, `original` and `restored`, initially hold the same invented capability. A pre-agreed recovery authority can replace the credential and epoch for the `research` resource under two explicitly approved transitions. The separate `archive` resource, historic attribution and the unresolved state of partnership consent remain outside those grants.

```sh
python3 model/return_model.py
python3 -m unittest discover -s tests -v
make check
```

The CLI emits deterministic JSON to stdout and never writes a file itself. To deliberately regenerate the committed default-bound artifact:

```sh
python3 model/return_model.py > evidence/return-model-results.json
```

The default bound is four events; `--bound 1` through `--bound 6` are supported. Python's standard library is sufficient. The source is [return_model.py](return_model.py); the artifact is [return-model-results.json](../evidence/return-model-results.json).

## What is trusted inside the experiment

Every credential is a synthetic string. A stipulated possession relation constrains which actor can present it. Knowing a string in this source does not grant simulated possession. This models an unforgeable capability **by assumption**, without real cryptography. The actor label is ground truth for the experiment's counterexample observer; a real resource checker does not learn an actor's identity from it.

The checker uses the presented credential and epoch. It ignores names and branch claims. A request marked `impossible_without_possession` was an attempted scenario the possession oracle excludes; it is not an additional authentication mechanism available to a real server. A holder of a shared or stolen current credential passes the actual checker regardless of claimed name.

The recovery grants pre-exist this model. They authorize only research epoch `0 → 1` with replacement capability 1, and epoch `1 → 2` with replacement capability 2. Grant legitimacy, authority appointment and replacement-recipient selection are unresolved policy inputs. The `missing_recovery_policy` variant removes those grants; it reports pending recovery and preserves existing access rather than inventing a freeze or operator power.

All actions use one immutable, serialized state machine. In the coherent variant, replacement updates both the resource record and its modeled effect checker atomically. Every modeled resource action consults that current state. This is a strong enforcement assumption, **not a demonstrated distributed implementation**. Already completed effects are not reversed. Replays and reordering of the two approved recovery operations are modeled; payment replay protection is not.

## Findings within the default bound

| Variant | Distinct states, including initial | Observation over all attempt prefixes |
|---|---:|---|
| Atomic replacement, current checker, expected-epoch comparison | 3 | No old credential accepted after recovery; no recovery applied at an unexpected epoch |
| Stale checker retains old epoch and credential | 3 | 1,980 prefixes end in old-credential acceptance after recovery; a direct control rejects the replacement credential |
| Expected-epoch comparison omitted | 3 | 2,838 prefixes apply recovery at an unexpected current epoch; 458 roll the resource epoch backward |
| Original actor also possesses replacement capability 1 | 3 | 457 prefixes allow that actor after recovery using the shared replacement |
| No pre-agreed recovery grants | 1 | Recovery remains pending; existing authorizations remain usable |

These are model counts, not probabilities, observed incidents or success rates. The artifact includes shortest observed counterexample sequences. The stale-checker example is simply approved recovery followed by an old-credential request. In the no-comparison variant, the second grant can arrive first; applying the earlier grant afterward rolls the epoch back. A normal `0 → 1 → 2` chain followed by a duplicate of the first grant also rolls back without the comparison. The tests cover both the required ordering and that three-step replay.

Before recovery, both actors can act using the same old capability; changing the claimed name cannot distinguish them. After coherent recovery, a current replacement credential succeeds even under an arbitrary claimed name. The old credential fails even if its holder claims to be `restored` and submits the new epoch. The experiment does not confuse denying everyone with secure recovery.

The shared-replacement control qualifies the conclusion: replacing a credential excludes old-only credential holders. It cannot establish one independently controlled participant when a current credential is copied, stolen or jointly held. Likewise, a record of a new epoch cannot fence an old credential at a downstream checker that still trusts the earlier state.

History and archive access stay unchanged under the research recovery grant. Partnership consent stays `undetermined`; no credential transition manufactures agreement to continue a partnership. The attribution field only demonstrates that this transition leaves an unrelated record untouched, not that the model implements an attribution protocol.

## Exhaustive scope and counting

The fixed alphabet contains 13 events: ten resource requests, two approved recoveries and one request for recovery made using the old capability under a claimed authority name. There are two resources, three simulated actors, two replacement capabilities and at most four events per sequence in the committed result.

For **each** variant the search covers `13 + 13² + 13³ + 13⁴ = 30,940` nonempty attempt-sequence prefixes, including 28,561 sequences of exactly four events. Denied requests and possession-impossible attempts are counted as no-change outcomes. These counts therefore describe all words in the attempt alphabet, not only feasible authenticated execution traces. Outcome counters sum to that same prefix total.

Equivalent states are merged while retaining their sequence multiplicities. The search performs 117 state/event evaluations after per-depth merging for each of the atomic, stale-checker and shared-replacement variants; 130 for the no-comparison variant; and 52 for missing policy. The same state/event pair can be evaluated at more than one depth. State counts are local to each configuration. The result records per-depth state counts and counterexample witnesses, so a large attempted-sequence count is not mistaken for a large state space.

The weighted observation count records prefixes **ending** in the described event. It does not count all longer sequences containing an earlier failure. None of this proves behavior for arbitrary event alphabets or unbounded traces.

## Limits and falsifier

This is a bounded authorization experiment. It does not model distributed consensus, real cache invalidation, network partitions, race conditions inside a resource action, process isolation, key loss, later compromise, recovery-authority compromise, liveness, independent controllers, coercion, a funds ledger, compensation for earlier effects, or subjective identity. Replacement credentials have fixed possession sets from initialization; secure handoff is assumed, not simulated.

Within the atomic variant's assumptions, a trace that permits an old-only holder after recovery, lets an unapproved recovery change the resource, or re-applies a grant at an unexpected epoch would overturn the reported conditional result. The included counterexamples show exactly why stale checkers, shared replacement credentials and missing expected-epoch comparison fall outside those assumptions.

Before applying the idea to actual software, identify the owner-authorized recovery rule and every place where the resulting resource action takes effect. An unmodeled checker that accepts old authority would falsify the claim that the deployed system satisfies this model's current-checker assumption. That inspection would be a new source/runtime investigation; this repository makes no live change.
