"""Policy-boundary tests for an offline symbolic experiment."""

from dataclasses import replace
import unittest

from model.return_model import (
    AUTHORITY, NEW1, OLD, Configuration, Recovery, Request,
    alphabet, explore, initial, recovery, step,
)


class ReturnStudyTests(unittest.TestCase):
    def setUp(self):
        self.configuration = Configuration("test")
        self.before = initial(self.configuration)
        self.after, outcome = step(self.before, recovery(), self.configuration)
        self.assertEqual(outcome, "recovery_applied")

    def request(self, actor="original", claim="original", credential=OLD,
                epoch=0, resource="research"):
        return Request(actor, claim, resource, credential, epoch)

    def test_shared_old_capability_cannot_distinguish_actors(self):
        for actor in ("original", "restored"):
            for claim in ("original", "restored", "founder", ""):
                with self.subTest(actor=actor, claim=claim):
                    self.assertEqual(step(self.before, self.request(actor, claim),
                                          self.configuration)[1], "allow")

    def test_old_holder_cannot_bypass_rotation_by_relabeling_or_epoch_change(self):
        for claim in ("original", "restored", "recovery-authority", ""):
            for epoch in (0, 1, 2):
                with self.subTest(claim=claim, epoch=epoch):
                    self.assertEqual(step(self.after, self.request(claim=claim, epoch=epoch),
                                          self.configuration)[1], "reject_checker")

    def test_current_capability_is_a_positive_control_independent_of_name(self):
        for claim in ("original", "restored", "any-name"):
            event = self.request("restored", claim, NEW1, 1)
            self.assertEqual(step(self.after, event, self.configuration)[1], "allow")

    def test_knowing_capability_label_does_not_create_possession_in_model(self):
        event = self.request("original", "restored", NEW1, 1)
        self.assertEqual(step(self.after, event, self.configuration)[1],
                         "impossible_without_possession")

    def test_shared_or_stolen_replacement_defeats_actor_exclusivity(self):
        configuration = Configuration("shared", original_holds_replacement=True)
        state, _ = step(initial(configuration), recovery(), configuration)
        event = self.request("original", "original", NEW1, 1)
        self.assertEqual(step(state, event, configuration)[1], "allow")

    def test_stale_checker_accepts_revoked_credential_and_rejects_current(self):
        configuration = Configuration("stale", coherent_checker=False)
        state, _ = step(initial(configuration), recovery(), configuration)
        self.assertEqual(state.resource("research").epoch, 1)
        self.assertEqual(step(state, self.request(), configuration)[1], "allow")
        current = self.request("restored", "restored", NEW1, 1)
        self.assertEqual(step(state, current, configuration)[1], "reject_checker")

    def test_future_recovery_requires_expected_current_epoch(self):
        state, outcome = step(self.before, recovery(1), self.configuration)
        self.assertEqual(outcome, "reject_expected_epoch")
        self.assertEqual(state, self.before)

    def test_duplicate_and_old_recovery_do_not_apply_again(self):
        duplicate, outcome = step(self.after, recovery(0), self.configuration)
        self.assertEqual((duplicate, outcome), (self.after, "reject_expected_epoch"))
        second, outcome = step(self.after, recovery(1), self.configuration)
        self.assertEqual(outcome, "recovery_applied")
        replay, outcome = step(second, recovery(0), self.configuration)
        self.assertEqual((replay, outcome), (second, "reject_expected_epoch"))

    def test_omitting_epoch_comparison_allows_reordered_recovery_rollback(self):
        configuration = Configuration("no-cas", compare_expected_epoch=False)
        state = initial(configuration)
        for event in (recovery(0), recovery(1), recovery(0)):
            state, outcome = step(state, event, configuration)
            self.assertEqual(outcome, "recovery_applied")
        self.assertEqual(state.resource("research").epoch, 1)
        self.assertEqual(step(state, self.request("restored", "restored", NEW1, 1),
                              configuration)[1], "allow")

    def test_missing_policy_is_pending_and_does_not_invent_freeze_or_authority(self):
        configuration = Configuration("pending", has_recovery_policy=False)
        before = initial(configuration)
        state, outcome = step(before, recovery(), configuration)
        self.assertEqual((state, outcome), (before, "pending_no_recovery_policy"))
        self.assertEqual(step(state, self.request(), configuration)[1], "allow")

    def test_claimed_authority_with_old_capability_cannot_recover(self):
        forged = replace(recovery(), actor="original", claimed_name="recovery-authority",
                         credential=OLD)
        state, outcome = step(self.before, forged, self.configuration)
        self.assertEqual((state, outcome), (self.before, "reject_recovery_authority"))

    def test_recovery_is_scoped_to_approved_resource_and_target(self):
        unapproved = (
            replace(recovery(), resource="archive"),
            replace(recovery(), replacement=OLD),
        )
        for event in unapproved:
            state, outcome = step(self.before, event, self.configuration)
            self.assertEqual((state, outcome), (self.before, "reject_unapproved_recovery"))

    def test_other_resources_history_and_consent_are_not_reassigned(self):
        self.assertEqual(self.before.resource("archive"), self.after.resource("archive"))
        self.assertEqual(step(self.after, self.request(resource="archive"),
                              self.configuration)[1], "allow")
        self.assertEqual(self.after.attribution, self.before.attribution)
        self.assertEqual(self.after.partnership_consent, "undetermined")

    def test_bounded_exploration_reports_controls_and_exact_trace_accounting(self):
        bound = 3
        atomic = explore(self.configuration, bound)
        stale = explore(Configuration("stale", coherent_checker=False), bound)
        no_cas = explore(Configuration("no-cas", compare_expected_epoch=False), bound)
        shared = explore(Configuration("shared", original_holds_replacement=True), bound)
        self.assertEqual(atomic["nonempty_trace_prefixes"],
                         sum(len(alphabet()) ** depth for depth in range(1, bound + 1)))
        self.assertEqual(atomic["full_length_traces"], len(alphabet()) ** bound)
        self.assertEqual(sum(atomic["outcomes_over_all_prefixes"].values()),
                         atomic["nonempty_trace_prefixes"])
        self.assertEqual(atomic["observation_counts_over_all_prefixes"], {})
        self.assertGreater(stale["observation_counts_over_all_prefixes"]
                           ["old_credential_accepted_after_recovery"], 0)
        self.assertGreater(no_cas["observation_counts_over_all_prefixes"]
                           ["resource_epoch_rolled_back"], 0)
        self.assertGreater(shared["observation_counts_over_all_prefixes"]
                           ["original_actor_accepted_after_recovery"], 0)


if __name__ == "__main__":
    unittest.main()
