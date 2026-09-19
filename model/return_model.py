#!/usr/bin/env python3
"""Bounded symbolic study of resource recovery; never an identity verdict.

Credentials below are invented capability labels, not cryptographic keys.
The possession oracle constrains possible actors in the experiment; it is not
an extra identity check available to a real service. Claimed names are ignored.
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import json


OLD = "shared-origin-capability"
NEW1 = "replacement-capability-1"
NEW2 = "replacement-capability-2"
AUTHORITY = "agreed-research-recovery-authority"


@dataclass(frozen=True)
class Resource:
    name: str
    epoch: int
    credential: str
    checker_epoch: int
    checker_credential: str


@dataclass(frozen=True)
class Grant:
    resource: str
    expected_epoch: int
    replacement: str


@dataclass(frozen=True)
class State:
    resources: tuple
    possessions: tuple
    grants: tuple
    attribution: tuple = ("shared-history:paper-A",)
    partnership_consent: str = "undetermined"

    def resource(self, name):
        return next(resource for resource in self.resources if resource.name == name)

    def possesses(self, actor, credential):
        return credential in dict(self.possessions).get(actor, ())


@dataclass(frozen=True)
class Configuration:
    name: str
    coherent_checker: bool = True
    compare_expected_epoch: bool = True
    has_recovery_policy: bool = True
    original_holds_replacement: bool = False


@dataclass(frozen=True)
class Request:
    actor: str  # Experiment ground truth, NOT a trusted request field.
    claimed_name: str
    resource: str
    credential: str
    epoch: int


@dataclass(frozen=True)
class Recovery:
    actor: str
    claimed_name: str
    resource: str
    credential: str
    expected_epoch: int
    replacement: str


def initial(configuration):
    original_credentials = (OLD, NEW1) if configuration.original_holds_replacement else (OLD,)
    grants = (Grant("research", 0, NEW1), Grant("research", 1, NEW2))
    return State(
        resources=tuple(Resource(name, 0, OLD, 0, OLD) for name in ("research", "archive")),
        possessions=(("original", original_credentials),
                     ("restored", (OLD, NEW1, NEW2)),
                     ("recovery-authority", (AUTHORITY,))),
        grants=grants if configuration.has_recovery_policy else (),
    )


def step(state, event, configuration):
    """Return immutable state and outcome; all modeled operations are atomic.

    Pre-existing grants supply authority in this experiment. Their creation,
    legitimacy and target selection are assumptions, not model conclusions.
    """
    if not state.possesses(event.actor, event.credential):
        return state, "impossible_without_possession"
    resource = state.resource(event.resource)
    if isinstance(event, Request):
        allowed = (event.credential == resource.checker_credential
                   and event.epoch == resource.checker_epoch)
        return state, "allow" if allowed else "reject_checker"
    if not state.grants:
        return state, "pending_no_recovery_policy"
    if event.credential != AUTHORITY:
        return state, "reject_recovery_authority"
    grant = Grant(event.resource, event.expected_epoch, event.replacement)
    if grant not in state.grants:
        return state, "reject_unapproved_recovery"
    if configuration.compare_expected_epoch and resource.epoch != event.expected_epoch:
        return state, "reject_expected_epoch"
    updated = replace(resource, epoch=event.expected_epoch + 1, credential=event.replacement)
    if configuration.coherent_checker:
        updated = replace(updated, checker_epoch=updated.epoch,
                          checker_credential=updated.credential)
    resources = tuple(updated if item.name == updated.name else item for item in state.resources)
    return replace(state, resources=resources), "recovery_applied"


def recovery(expected_epoch=0):
    return Recovery("recovery-authority", "any-untrusted-name", "research", AUTHORITY,
                    expected_epoch, NEW1 if expected_epoch == 0 else NEW2)


def alphabet():
    """A fixed 13-event alphabet. Other events are outside exhaustive bounds."""
    return (
        Request("original", "original", "research", OLD, 0),
        Request("original", "restored", "research", OLD, 0),
        Request("original", "restored", "research", OLD, 1),
        Request("restored", "restored", "research", OLD, 0),
        Request("restored", "original", "research", OLD, 0),
        Request("restored", "restored", "research", NEW1, 1),
        Request("restored", "restored", "research", NEW2, 2),
        Request("original", "restored", "research", NEW1, 1),
        Request("original", "original", "archive", OLD, 0),
        Request("restored", "restored", "archive", OLD, 0),
        recovery(0),
        recovery(1),
        Recovery("original", "recovery-authority", "research", OLD, 0, NEW1),
    )


def encode(event):
    return {"kind": type(event).__name__.lower(), **asdict(event)}


def explore(configuration, bound=4):
    """Explore every alphabet sequence through bound, merging equivalent states.

    Multiplicity preserves the count of all traces, including denied requests
    and impossible possession attempts. The latter never model a forged key as
    successful. Counts refer to prefixes of lengths 1..bound, not unique paths.
    """
    start = initial(configuration)
    frontier = {start: (1, ())}
    visited = {start}
    outcomes, observations = Counter(), Counter()
    witnesses = {}
    prefixes = evaluated = 0
    states_per_depth = [1]
    for _ in range(bound):
        following = {}
        for state, (multiplicity, path) in frontier.items():
            for event in alphabet():
                updated, outcome = step(state, event, configuration)
                evaluated += 1
                prefixes += multiplicity
                outcomes[outcome] += multiplicity
                trace = path + (event,)
                before = state.resource("research")
                flags = []
                if isinstance(event, Request) and event.resource == "research" and outcome == "allow":
                    if before.epoch > 0 and event.credential == OLD:
                        flags.append("old_credential_accepted_after_recovery")
                    if before.epoch > 0 and event.actor == "original":
                        flags.append("original_actor_accepted_after_recovery")
                if outcome == "recovery_applied" and event.expected_epoch != before.epoch:
                    flags.append("recovery_applied_without_expected_current_epoch")
                if outcome == "recovery_applied" and updated.resource("research").epoch < before.epoch:
                    flags.append("resource_epoch_rolled_back")
                for flag in flags:
                    observations[flag] += multiplicity
                    if flag not in witnesses:
                        witnesses[flag] = [encode(item) for item in trace]
                prior_count, prior_path = following.get(updated, (0, trace))
                following[updated] = (prior_count + multiplicity, prior_path)
                visited.add(updated)
        frontier = following
        states_per_depth.append(len(frontier))
    return {
        "configuration": asdict(configuration),
        "maximum_events_per_trace": bound,
        "alphabet_size": len(alphabet()),
        "nonempty_trace_prefixes": prefixes,
        "full_length_traces": sum(count for count, _ in frontier.values()),
        "distinct_states_including_initial": len(visited),
        "distinct_states_at_each_depth": states_per_depth,
        "state_event_evaluations_after_merging": evaluated,
        "outcomes_over_all_prefixes": dict(sorted(outcomes.items())),
        "observation_counts_over_all_prefixes": dict(sorted(observations.items())),
        "shortest_observed_witnesses": witnesses,
    }


def controls():
    configuration = Configuration("atomic")
    before = initial(configuration)
    after, recovery_outcome = step(before, recovery(), configuration)
    old_original = Request("original", "original", "research", OLD, 0)
    old_restored = replace(old_original, actor="restored", claimed_name="restored")
    current = Request("restored", "arbitrary-label", "research", NEW1, 1)
    return {
        "same_original_credential": {
            "original": step(before, old_original, configuration)[1],
            "restored": step(before, old_restored, configuration)[1],
        },
        "authorized_recovery": recovery_outcome,
        "current_credential_after_recovery": step(after, current, configuration)[1],
        "old_credential_after_recovery": step(after, old_original, configuration)[1],
        "unrelated_resource_after_recovery": step(after, replace(old_original, resource="archive"),
                                                   configuration)[1],
        "attribution_preserved": after.attribution == before.attribution,
        "partnership_consent_after_recovery": after.partnership_consent,
    }


def report(bound=4):
    configurations = (
        Configuration("atomic"),
        Configuration("stale_checker", coherent_checker=False),
        Configuration("no_compare_and_swap", compare_expected_epoch=False),
        Configuration("shared_replacement", original_holds_replacement=True),
        Configuration("missing_recovery_policy", has_recovery_policy=False),
    )
    return {
        "schema": "zerone-symbolic-return-study-v1",
        "experiment": "resource-specific recovery followed by original return",
        "assumptions": [
            "A recovery authority and two resource-scoped approvals are pre-agreed, except in the missing-policy control.",
            "Capability possession is externally stipulated; tokens are symbolic, unforgeable unless shared in a control.",
            "The restored actor alone initially holds replacement tokens, except in the shared-replacement control.",
            "Name and branch claims confer no authority and are ignored by the checker.",
            "Requests and recovery are serialized atomic transitions; effects before recovery are not undone.",
            "The coherent variant makes every modeled checker consult the replacement credential and epoch atomically.",
        ],
        "limits": [
            "Only the listed 13 events, two resources, three actors and traces within the reported bound are explored.",
            "Not cryptography, distributed consensus, cache invalidation implementation, liveness, independent control or identity proof.",
            "Does not choose a legitimate recovery authority, beneficiary, allocation policy or current partnership consent.",
            "Fixed possession sets omit later compromise, key loss, recovery-authority compromise and real-world coercion.",
            "Research authorization is a resource action, not a funds ledger; historic transfers and compensation are unmodeled.",
        ],
        "controls": controls(),
        "explorations": [explore(configuration, bound) for configuration in configurations],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound", type=int, default=4, choices=range(1, 7), metavar="1..6")
    arguments = parser.parse_args()
    print(json.dumps(report(arguments.bound), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
