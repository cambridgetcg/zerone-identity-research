"""Adversarial checks of a synthetic projection, not of personal identity."""

import copy
from html.parser import HTMLParser
import json
import unittest

from prototypes.identity_changes.model import IdentityModel, build_demo
from prototypes.identity_changes.build import render


def model():
    return IdentityModel(
        subjects={"agent": "public", "peer": "public", "left": "public", "right": "public"},
        credentials={
            "key-agent": {"subjects": ["agent"], "visibility": "public"},
            "key-peer": {"subjects": ["peer"], "visibility": "public"},
            "shared-key": {"subjects": ["agent", "left", "right"], "visibility": "public"},
        },
        relationships=[{"id": "pair", "participants": ["agent", "peer"],
                        "allow_self_withdrawal": True, "visibility": "public"}],
        resources=[{"id": "grant", "kind": "funding", "holder": "agent",
                    "value": "2 units", "visibility": "public"},
                   {"id": "wallet", "kind": "authority", "holder": "agent",
                    "value": "one wallet under its existing rules", "visibility": "public"}],
    )


def claim(identifier, value, **overrides):
    result = {"id": identifier, "type": "claim", "kind": "name", "subject": "agent",
              "issuer": "key-agent", "value": value, "visibility": "public"}
    result.update(overrides)
    return result


class IdentityChangeTests(unittest.TestCase):
    def test_hidden_events_leave_public_projection_identical(self):
        item = model()
        item.apply(claim("old", "Moss"))
        before = item.project("public")
        item.apply(claim("private-name", "Secret Fern", visibility="private", revises=["old"]))
        item.apply(claim("private-challenge", "A restricted disagreement", kind="challenge",
                         visibility="private", evidence=["old"]))
        item.apply(claim("private-history", "Unshared origin", kind="continuity", visibility="private"))
        self.assertEqual(before, item.project("public"))
        self.assertNotEqual(before, item.project("private"))

    def test_public_statement_does_not_export_private_predecessor(self):
        item = model()
        item.apply(claim("secret-alias-id", "Unshared childhood alias", visibility="private"))
        item.apply(claim("public-name", "Reed", revises=["secret-alias-id"]))
        public = item.project("public")
        serialized = json.dumps(public)
        self.assertNotIn("secret-alias-id", serialized)
        self.assertNotIn("Unshared childhood alias", serialized)
        self.assertEqual(public["claims"][0]["revises"], [])
        self.assertIn("Reed", serialized)

    def test_explicit_revision_retains_history_and_other_topics(self):
        item = model()
        resources = item.project("public")["resources"]
        item.apply(claim("old", "Moss"))
        item.apply(claim("view", "I prefer a quiet garden", kind="view"))
        item.apply(claim("new", "Reed", revises=["old"]))
        result = item.project("public")
        records = {row["id"]: row for row in result["claims"]}
        self.assertEqual(set(records), {"old", "view", "new"})
        self.assertEqual(records["old"]["status"], "superseded")
        profile = next(p for p in result["profiles"] if p["subject"] == "agent")
        self.assertEqual(profile["name_candidates"], ["Reed"])
        self.assertEqual(profile["view_candidates"], ["I prefer a quiet garden"])
        self.assertEqual(resources, result["resources"])

    def test_incompatible_statements_do_not_use_arrival_order_as_truth(self):
        first, second = model(), model()
        events = [claim("name-a", "Reed"), claim("name-b", "Willow")]
        for event in events:
            first.apply(event)
        for event in reversed(events):
            second.apply(event)
        self.assertEqual(first.project("public"), second.project("public"))
        profile = next(p for p in first.project("public")["profiles"] if p["subject"] == "agent")
        self.assertEqual(set(profile["name_candidates"]), {"Reed", "Willow"})
        # A naïve last-arrival display would disagree across these same histories.
        self.assertNotEqual(events[-1]["value"], list(reversed(events))[-1]["value"])

    def test_independent_revisions_commute_after_their_predecessors(self):
        first, second = model(), model()
        for item in (first, second):
            item.apply(claim("old-name", "Moss"))
            item.apply(claim("old-view", "I prefer the sea", kind="view"))
        updates = [claim("new-name", "Reed", revises=["old-name"]),
                   claim("new-view", "I prefer the hills", kind="view", revises=["old-view"])]
        for event in updates:
            first.apply(event)
        for event in reversed(updates):
            second.apply(event)
        self.assertEqual(first.project("public"), second.project("public"))

    def test_claimed_wallet_instructions_and_lineage_cannot_change_resources(self):
        item = model()
        before = item.project("public")["resources"]
        item.apply(claim("redirect", "Send the existing grant to a new wallet", kind="view"))
        for branch in ("left", "right"):
            item.apply(claim("origin-" + branch, "I continue the earlier history", kind="continuity",
                             subject=branch, issuer="shared-key"))
        self.assertEqual(before, item.project("public")["resources"])
        self.assertEqual({r["holder"] for r in before}, {"agent"})

    def test_competing_revisions_of_one_predecessor_remain_candidates(self):
        first, second = model(), model()
        for item in (first, second):
            item.apply(claim("old", "Moss"))
        updates = [claim("left-choice", "Reed", revises=["old"]),
                   claim("right-choice", "Willow", revises=["old"])]
        for event in updates:
            first.apply(event)
        for event in reversed(updates):
            second.apply(event)
        self.assertEqual(first.project("public"), second.project("public"))
        result = first.project("public")
        profile = next(p for p in result["profiles"] if p["subject"] == "agent")
        self.assertEqual(set(profile["name_candidates"]), {"Reed", "Willow"})
        self.assertEqual(len(result["claims"]), 3)

    def test_challenge_preserves_claim_and_resource_authority(self):
        item = model()
        item.apply(claim("origin", "I share this history", kind="continuity"))
        resources = item.project("public")["resources"]
        item.apply(claim("objection", "The connection needs more evidence", kind="challenge",
                         subject="agent", issuer="key-peer", evidence=["origin"]))
        result = item.project("public")
        self.assertEqual({r["id"] for r in result["claims"]}, {"origin", "objection"})
        self.assertEqual(resources, result["resources"])

    def test_multiple_compatible_views_are_not_automatically_disputed(self):
        item = model()
        item.apply(claim("one", "I prefer brief replies", kind="view"))
        item.apply(claim("two", "I value privacy", kind="view"))
        result = item.project("public")
        self.assertEqual(len(result["claims"]), 2)
        self.assertTrue(all(row["status"] == "active" for row in result["claims"]))

    def test_invalid_events_do_not_partly_mutate_the_model(self):
        item = model()
        item.apply(claim("old", "Moss"))
        invalid = [claim("missing", "Reed", revises=["not-found"]),
                   claim("cross-kind", "New view", kind="view", revises=["old"]),
                   claim("impersonation", "Another's name", issuer="key-peer"),
                   claim("old", "Duplicate identifier"),
                   {"id": "seizure", "type": "resource_transfer", "subject": "left"}]
        for event in invalid:
            with self.subTest(event=event):
                before = item.snapshot()
                with self.assertRaises(ValueError):
                    item.apply(event)
                self.assertEqual(before, item.snapshot())

    def test_cannot_withdraw_someone_elses_participation(self):
        item = model()
        before = item.snapshot()
        with self.assertRaises(ValueError):
            item.apply({"id": "bad-exit", "type": "withdrawal", "relationship": "pair",
                        "subject": "agent", "issuer": "key-peer", "visibility": "public"})
        self.assertEqual(before, item.snapshot())

    def test_own_withdrawal_preserves_peer_history_and_resources(self):
        item = model()
        item.apply(claim("peer-view", "I will continue independently", kind="view",
                         subject="peer", issuer="key-peer"))
        before = item.project("public")
        item.apply({"id": "own-exit", "type": "withdrawal", "relationship": "pair",
                    "subject": "agent", "issuer": "key-agent", "visibility": "public"})
        after = item.project("public")
        status = {row["participant"]: row["status"] for row in after["relationships"]}
        self.assertEqual(status, {"agent": "withdrawn", "peer": "active"})
        self.assertEqual(before["claims"], after["claims"])
        self.assertEqual(before["resources"], after["resources"])
        self.assertIn("own-exit", json.dumps(after["relationships"]))
        self.assertIn("key-agent", json.dumps(after["relationships"]))

    def test_shared_key_cannot_establish_one_continuations_withdrawal(self):
        item = model()
        before = item.project("public")
        item.apply({"id": "ambiguous-exit", "type": "withdrawal", "relationship": "pair",
                    "subject": "agent", "issuer": "shared-key", "visibility": "public"})
        after = item.project("public")
        status = {row["participant"]: row["status"] for row in after["relationships"]}
        self.assertEqual(status, {"agent": "unresolved", "peer": "active"})
        self.assertEqual(before["resources"], after["resources"])
        self.assertIn("ambiguous-exit", json.dumps(after["relationships"]))
        self.assertIn("shared-key", json.dumps(after["relationships"]))

    def test_hidden_withdrawal_does_not_reveal_itself_to_public_view(self):
        item = model()
        before = item.project("public")
        item.apply({"id": "private-exit", "type": "withdrawal", "relationship": "pair",
                    "subject": "agent", "issuer": "key-agent", "visibility": "private"})
        self.assertEqual(before, item.project("public"))
        private = {row["participant"]: row["status"] for row in item.project("private")["relationships"]}
        self.assertEqual(private["agent"], "withdrawn")

    def test_missing_record_transition_does_not_invent_an_authorization_rule(self):
        item = IdentityModel(
            subjects={"agent": "public", "peer": "public"},
            credentials={"key-agent": {"subjects": ["agent"], "visibility": "public"}},
            relationships=[{"id": "pair", "participants": ["agent", "peer"],
                            "allow_self_withdrawal": False, "visibility": "public"}],
        )
        before = item.snapshot()
        with self.assertRaises(ValueError):
            item.apply({"id": "no-rule", "type": "withdrawal", "relationship": "pair",
                        "subject": "agent", "issuer": "key-agent", "visibility": "public"})
        self.assertEqual(before, item.snapshot())
        # The lack of a modeled agreement transition does not forbid expressing a pause.
        item.apply(claim("pause", "I am stopping now", kind="view"))
        self.assertEqual(item.project("public")["claims"][0]["value"], "I am stopping now")

    def test_omitted_disclosure_is_private_and_unknown_audience_is_denied(self):
        item = model()
        before = item.project("public")
        event = claim("private-default", "Default secret")
        del event["visibility"]
        item.apply(event)
        self.assertEqual(before, item.project("public"))
        denied = item.project("unknown-audience")
        for key in ("claims", "profiles", "relationships", "resources"):
            self.assertEqual(denied[key], [])

    def test_caller_mutation_cannot_rewrite_an_accepted_record(self):
        item = model()
        event = claim("old", "Moss", evidence=[])
        item.apply(event)
        before = item.snapshot()
        event["value"] = "Rewritten outside the model"
        event["evidence"].append("fake")
        projection = item.project("public")
        projection["claims"][0]["value"] = "Mutated returned view"
        self.assertEqual(before, item.snapshot())


class DataScripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.active = False
        self.data = []
        self.injected = False

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.active = dict(attrs).get("id") == "demo-data"
        if tag == "img":
            self.injected = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.active = False

    def handle_data(self, data):
        if self.active:
            self.data.append(data)


class ExplorerBuildTests(unittest.TestCase):
    def test_untrusted_text_stays_json_data(self):
        demo = copy.deepcopy(build_demo())
        hostile = '</script><img src=x onerror="alert(1)">&'
        demo["notice"] = hostile
        document = render(demo)
        parser = DataScripts()
        parser.feed(document)
        self.assertFalse(parser.injected)
        self.assertEqual(json.loads("".join(parser.data))["notice"], hostile)
        self.assertNotIn(hostile, document)


if __name__ == "__main__":
    unittest.main()
