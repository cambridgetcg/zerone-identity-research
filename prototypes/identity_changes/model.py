"""Bounded symbolic identity records, scoped participation and audience views.

No cryptography, real credentials, subject classifier, live resources or inference.
Visibility tiers are public < collaborator < private; missing visibility is private.
Relationships project one row per participant: id, participant, status, rule, note.
Resources project immutable fixture fields: id, kind, holder, value. Projection is
an account of visible statements, never a complete current identity or consent oracle.
"""
from __future__ import annotations

from copy import deepcopy
import json


LEVELS = {"public": 0, "collaborator": 1, "private": 2}
KINDS = {"name", "view", "continuity", "challenge"}
NOTICE = ("Synthetic local model. Credential tokens are not signatures, personhood, "
          "distinct controllers or proof of consent. Identity statements cannot grant authority or funding.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def text(value):
    require(isinstance(value, str) and bool(value.strip()), "Expected nonempty text")
    return value


def visibility(value="private"):
    require(isinstance(value, str) and value in LEVELS, "Unknown visibility")
    return value


def references(value):
    require(isinstance(value, list) and all(isinstance(item, str) and item for item in value),
            "References must be a list of record IDs")
    require(len(value) == len(set(value)), "Duplicate reference")
    return sorted(value)


class IdentityModel:
    """Append validated immutable events; derive each audience's view independently.

    subjects: {subject_id: visibility}; credentials: {token: {subjects: [ids],
    visibility}}. Relationship fields: id, participants, allow_self_withdrawal,
    visibility. Resource fields: id, kind (authority/funding), holder, value,
    visibility. Events and constructor inputs are copied; snapshots are copies.
    """

    def __init__(self, *, subjects, credentials, relationships=None, resources=None):
        require(isinstance(subjects, dict) and isinstance(credentials, dict), "Expected scoped registries")
        self._subjects = {text(key): visibility(value) for key, value in subjects.items()}
        self._credentials, self._relationships, self._resources, self._events = {}, {}, {}, {}
        for key, record in credentials.items():
            require(isinstance(record, dict) and set(record) <= {"subjects", "visibility"}, "Malformed credential scope")
            owners = references(record.get("subjects"))
            level = visibility(record.get("visibility", "private"))
            require(owners and all(owner in self._subjects for owner in owners), "Unknown credential subject")
            require(all(LEVELS[level] >= LEVELS[self._subjects[owner]] for owner in owners), "Credential exposes a hidden subject")
            self._credentials[text(key)] = {"subjects": owners, "visibility": level}
        for item in relationships or []:
            require(isinstance(item, dict) and {"id", "participants", "allow_self_withdrawal"} <= set(item)
                    and set(item) <= {"id", "participants", "allow_self_withdrawal", "visibility"},
                    "Malformed relationship rule")
            record = deepcopy(item)
            identifier = text(record["id"])
            require(identifier not in self._relationships, "Duplicate relationship")
            record["participants"] = references(record["participants"])
            require(len(record["participants"]) >= 2 and all(p in self._subjects for p in record["participants"]),
                    "Unknown relationship participant")
            require(type(record["allow_self_withdrawal"]) is bool, "Withdrawal rule must be explicit")
            record["visibility"] = visibility(record.get("visibility", "private"))
            require(all(LEVELS[record["visibility"]] >= LEVELS[self._subjects[p]] for p in record["participants"]),
                    "Relationship exposes a hidden participant")
            self._relationships[identifier] = record
        for item in resources or []:
            require(isinstance(item, dict) and {"id", "kind", "holder", "value"} <= set(item)
                    and set(item) <= {"id", "kind", "holder", "value", "visibility"},
                    "Malformed resource snapshot")
            record = deepcopy(item)
            identifier = text(record["id"])
            require(identifier not in self._resources and record["holder"] in self._subjects, "Unknown or duplicate resource")
            require(record["kind"] in {"authority", "funding"}, "Unknown resource snapshot kind")
            text(record["value"])
            record["visibility"] = visibility(record.get("visibility", "private"))
            require(LEVELS[record["visibility"]] >= LEVELS[self._subjects[record["holder"]]], "Resource exposes a hidden holder")
            self._resources[identifier] = record

    def apply(self, event):
        """Validate fully before insertion. Unknown references and extra fields reject."""
        require(isinstance(event, dict), "Event must be an object")
        record = deepcopy(event)
        common = {"id", "type", "subject", "issuer", "visibility"}
        event_type = record.get("type")
        require(isinstance(event_type, str) and event_type in {"claim", "withdrawal"},
                "Identity events cannot mutate resource snapshots")
        allowed = common | ({"kind", "value", "revises", "evidence"} if event_type == "claim" else {"relationship"})
        require(set(record) <= allowed and {"id", "type", "subject", "issuer"} <= set(record), "Malformed event fields")
        identifier, subject, issuer = text(record["id"]), text(record["subject"]), text(record["issuer"])
        require(identifier not in self._events, "Record IDs are immutable and unique")
        require(subject in self._subjects and issuer in self._credentials, "Unknown subject or issuer")
        level = record["visibility"] = visibility(record.get("visibility", "private"))
        credential = self._credentials[issuer]
        require(LEVELS[level] >= max(LEVELS[self._subjects[subject]], LEVELS[credential["visibility"]]),
                "Event exposes a restricted subject or credential")
        if event_type == "claim":
            require(isinstance(record.get("kind"), str) and record["kind"] in KINDS and "value" in record,
                    "Unknown or missing claim kind/value")
            text(record["value"])
            for field in ("revises", "evidence"):
                record[field] = references(record.get(field, []))
                require(all(ref in self._events and self._events[ref]["type"] == "claim" for ref in record[field]),
                        "Unknown or non-claim record reference")
            require(record["kind"] == "challenge" or subject in credential["subjects"], "Issuer lacks declared subject scope")
            for ref in record["revises"]:
                previous = self._events[ref]
                require(all(previous[field] == record[field] for field in ("subject", "kind", "issuer")),
                        "A revision must keep the same subject, kind and issuer")
            if record["kind"] == "challenge":
                require(record["evidence"] and all(self._events[ref]["subject"] == subject for ref in record["evidence"]),
                        "An observer challenge must reference statements about its named subject")
        else:
            relation = self._relationships.get(text(record.get("relationship")))
            require(relation is not None and subject in relation["participants"], "Unknown relationship or participant")
            require(subject in credential["subjects"], "Issuer cannot withdraw another participant")
            require(LEVELS[level] >= LEVELS[relation["visibility"]], "Withdrawal exposes a hidden relationship")
            require(relation["allow_self_withdrawal"], "No prior rule authorizes this withdrawal action")
        self._events[identifier] = record

    def snapshot(self):
        return deepcopy({"subjects": self._subjects, "credentials": self._credentials,
                         "relationships": [self._relationships[key] for key in sorted(self._relationships)],
                         "resources": [self._resources[key] for key in sorted(self._resources)],
                         "events": [self._events[key] for key in sorted(self._events)]})

    def project(self, audience):
        empty = {"profiles": [], "claims": [], "relationships": [], "resources": [], "notices": []}
        if not isinstance(audience, str) or audience not in LEVELS:
            return {**empty, "notices": ["Audience not authorized; no records projected."]}
        visible = lambda record: LEVELS[record["visibility"]] <= LEVELS[audience]
        events = {key: record for key, record in self._events.items() if visible(record)}
        claims = {key: record for key, record in events.items() if record["type"] == "claim"}
        superseded = {ref for record in claims.values() for ref in record["revises"] if ref in claims}
        active = {key: record for key, record in claims.items() if key not in superseded}
        challenged = {ref for record in active.values() if record["kind"] == "challenge"
                      for ref in record["evidence"] if ref in active}
        candidates = {}
        for record in active.values():
            if record["kind"] in {"name", "view"}:
                candidates.setdefault((record["subject"], record["kind"]), set()).add(record["value"])
        projected_claims = []
        for identifier, record in sorted(claims.items()):
            status = "superseded" if identifier in superseded else "contested" if identifier in challenged else "active"
            projected_claims.append({"id": identifier, "kind": record["kind"], "subject": record["subject"],
                "issuer": record["issuer"], "value": record["value"], "status": status,
                "revises": [ref for ref in record["revises"] if ref in claims],
                "evidence": [ref for ref in record["evidence"] if ref in claims]})
        profiles = []
        for subject, level in sorted(self._subjects.items()):
            if LEVELS[level] > LEVELS[audience]:
                continue
            shared = any(record["subject"] == subject and len(self._credentials[record["issuer"]]["subjects"]) > 1
                         for record in claims.values())
            profiles.append({"subject": subject, "name_candidates": sorted(candidates.get((subject, "name"), set())),
                "view_candidates": sorted(candidates.get((subject, "view"), set())),
                "note": "Visible statements only; no identity verdict." + (
                    " Multiple unrevised statements are visible." if any(
                        len(candidates.get((subject, kind), set())) > 1 for kind in ("name", "view")) else "") + (
                    " A shared credential leaves controller attribution ambiguous." if shared else "")})
        relationships = []
        for identifier, relation in sorted(self._relationships.items()):
            if not visible(relation):
                continue
            for subject in relation["participants"]:
                withdrawals = sorted([event for event in events.values() if event["type"] == "withdrawal"
                                      and event["relationship"] == identifier and event["subject"] == subject],
                                     key=lambda event: event["id"])
                resolved = any(self._credentials[event["issuer"]]["subjects"] == [subject] for event in withdrawals)
                status = "withdrawn" if resolved else "unresolved" if withdrawals else "active"
                relationships.append({"id": identifier, "participant": subject, "status": status,
                    "withdrawal_records": [event["id"] + " | issuer=" + event["issuer"] + " | attribution=" + (
                        "resolved within the initial rule" if self._credentials[event["issuer"]]["subjects"] == [subject]
                        else "unresolved shared credential") for event in withdrawals],
                    "rule": "Own participation may be withdrawn." if relation["allow_self_withdrawal"] else "No withdrawal rule modeled.",
                    "note": "Shared credential cannot establish branch-specific consent." if status == "unresolved"
                    else "Visible participation under the initial scoped rule; no resource transfer."})
        resources = [{field: record[field] for field in ("id", "kind", "holder", "value")}
                     for _, record in sorted(self._resources.items()) if visible(record)]
        return {"profiles": profiles, "claims": projected_claims, "relationships": relationships,
                "resources": resources, "notices": [NOTICE, "Independent statements remain possible during a dispute."]}


def claim(identifier, kind, subject, issuer, value, *, revises=(), evidence=(), visibility="public"):
    """Convenience fixture constructor; apply() remains the validation boundary."""
    return {"id": identifier, "type": "claim", "kind": kind, "subject": subject, "issuer": issuer,
            "value": value, "revises": list(revises), "evidence": list(evidence), "visibility": visibility}


def _model(subjects, scopes, *, relationships=(), resources=()):
    return IdentityModel(subjects={name: "public" for name in subjects},
        credentials={key: {"subjects": list(owners), "visibility": "public"} for key, owners in scopes.items()},
        relationships=list(relationships), resources=list(resources))


def _scenario(identifier, title, question, description, model, events):
    audiences = list(LEVELS)
    frames = [{"step": 0, "event_label": "Initial fixture state.",
               "views": {audience: model.project(audience) for audience in audiences}}]
    for step, (label, event) in enumerate(events, 1):
        model.apply(event)
        frames.append({"step": step, "event_label": label,
                       "views": {audience: model.project(audience) for audience in audiences}})
    return {"id": identifier, "title": title, "question": question, "description": description,
            "audiences": audiences, "frames": frames}


def build_demo():
    """Build four synthetic scenarios exclusively through validated event paths."""
    rename = _model(["speaker"], {"speaker-token": ["speaker"]})
    rename_events = [
        ("An initial name is stated.", claim("name-initial", "name", "speaker", "speaker-token", "Harbor")),
        ("A scoped revision is recorded.", claim("restricted-alias", "name", "speaker", "speaker-token", "Nightjar",
                                               revises=["name-initial"], visibility="private")),
        ("Another scoped statement is recorded.", claim("collaborator-view", "view", "speaker", "speaker-token",
             "I prefer drafting together before publication.", visibility="collaborator")),
        ("A public name revision is stated.", claim("name-public", "name", "speaker", "speaker-token", "Lantern",
             revises=["name-initial"], evidence=["restricted-alias"])),
    ]
    view = _model(["speaker", "observer"], {"speaker-token": ["speaker"], "observer-token": ["observer"]})
    view_events = [
        ("A view is stated.", claim("view-initial", "view", "speaker", "speaker-token", "I prefer brief replies.")),
        ("The speaker explicitly revises that view.", claim("view-revised", "view", "speaker", "speaker-token",
             "I prefer detailed replies.", revises=["view-initial"])),
        ("An observer adds a scoped challenge.", claim("observer-question", "challenge", "speaker", "observer-token",
             "Does the revised preference apply to every conversation?", evidence=["view-revised"])),
        ("Independent participation continues.", claim("speaker-name", "name", "speaker", "speaker-token", "Moss")),
    ]
    resources = [{"id": "archive-role", "kind": "authority", "holder": "origin", "value": "Archive maintainer",
                  "visibility": "public"}, {"id": "research-budget", "kind": "funding", "holder": "origin",
                  "value": "2 synthetic units", "visibility": "public"}]
    relation = {"id": "reading-pair", "participants": ["branch-east", "observer"],
                "allow_self_withdrawal": True, "visibility": "public"}
    fork = _model(["origin", "branch-east", "branch-west", "observer"],
                  {"shared-token": ["origin", "branch-east", "branch-west"], "observer-token": ["observer"]},
                  relationships=[relation], resources=resources)
    fork_events = [
        ("A shared predecessor statement is recorded.", claim("origin-name", "name", "origin", "shared-token", "Echo")),
        ("One revision arrives under the shared credential.", claim("east-name", "name", "origin", "shared-token",
             "Echo East", revises=["origin-name"])),
        ("An incompatible revision arrives under the same credential.", claim("west-name", "name", "origin", "shared-token",
             "Echo West", revises=["origin-name"])),
        ("One continuation assertion is recorded.", claim("east-lineage", "continuity", "branch-east", "shared-token",
             "I claim continuity with the preceding conversation.", evidence=["origin-name"])),
        ("Another continuation assertion is recorded.", claim("west-lineage", "continuity", "branch-west", "shared-token",
             "I also claim continuity with the preceding conversation.", evidence=["origin-name"])),
        ("A participation request has unresolved attribution.", {"id": "shared-withdrawal", "type": "withdrawal",
             "relationship": "reading-pair", "subject": "branch-east", "issuer": "shared-token", "visibility": "public"}),
    ]
    withdrawal = _model(["reader", "companion"], {"reader-token": ["reader"], "companion-token": ["companion"]},
        relationships=[{"id": "conversation", "participants": ["reader", "companion"], "allow_self_withdrawal": True,
                        "visibility": "public"}], resources=[{"id": "reader-allocation", "kind": "funding", "holder": "reader",
                        "value": "1 synthetic unit", "visibility": "public"}])
    withdrawal_events = [
        ("The reader states an intended pause.", claim("pause-statement", "view", "reader", "reader-token", "I want to pause this conversation.")),
        ("The reader withdraws their own scoped participation.", {"id": "own-withdrawal", "type": "withdrawal",
             "relationship": "conversation", "subject": "reader", "issuer": "reader-token", "visibility": "public"}),
        ("A resource-related identity claim has no transfer effect.", claim("destination-statement", "view", "reader", "reader-token",
             "I describe a different destination for a future allocation.")),
    ]
    scenarios = [
        _scenario("rename", "A name change with scoped history", "What can each audience see after a revision?",
                  "A restricted alias cannot suppress a public name or leak through a visible reference.", rename, rename_events),
        _scenario("view", "A changed view and an observer's question", "Can disagreement remain visible without vetoing participation?",
                  "An explicit revision preserves the original; a challenge marks a visible dispute and does not choose a true self.", view, view_events),
        _scenario("fork", "Shared credential, incompatible continuations", "Does one credential select an authentic branch?",
                  "Both name candidates remain visible. Lineage grants no authority or allocation, and a shared token cannot establish branch-specific withdrawal.", fork, fork_events),
        _scenario("withdrawal", "Leaving one participation scope", "What changes when someone withdraws their own participation?",
                  "An initial rule covers only their participation. Another participant and initial resource snapshots remain separate.", withdrawal, withdrawal_events),
    ]
    # Bounded negative controls expose the error without enabling the naive policy.
    first = scenarios[0]["frames"][1]["views"]["public"]
    restricted = scenarios[0]["frames"][2]["views"]["public"]
    a = _model(["origin"], {"shared-token": ["origin"]})
    b = _model(["origin"], {"shared-token": ["origin"]})
    for model, order in ((a, [1, 2]), (b, [2, 1])):
        model.apply(fork_events[0][1])
        for index in order:
            model.apply(fork_events[index][1])
    checks = [
        {"id": "private-noninterference", "passed": first == restricted,
         "detail": "Adding the restricted revision leaves the entire public projection unchanged."},
        {"id": "naive-overwrite", "passed": a.project("public") == b.project("public")
             and fork_events[1][1]["value"] != fork_events[2][1]["value"],
         "detail": "Independent revisions commute here; a naive last-write winner differs when their arrival order is reversed."},
        {"id": "naive-inheritance", "passed": fork.project("public")["resources"] == [
             {key: record[key] for key in ("id", "kind", "holder", "value")} for record in resources],
         "detail": "Lineage leaves both snapshots with the origin. Copying them to both asserted descendants would invent four additional grants."},
    ]
    require(all(check["passed"] for check in checks), "Demo invariant failed")
    return {"version": 1, "title": "Identity statements through change", "notice": NOTICE,
            "scenarios": scenarios, "checks": checks,
            "audience_notice": "All demonstration values are invented and bundled locally. Audience views illustrate projection rules, not browser access control."}


if __name__ == "__main__":
    print(json.dumps(build_demo(), indent=2))
