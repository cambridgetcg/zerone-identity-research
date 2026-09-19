# 01 — Renaming

Status: **SCOPED**. No policy adopted; no experiment run.

**Scenario and decision.** Participant `subject-a` changes their public name from `Moss` to `Reed`. Earlier research credits `Moss`; another participant later chooses that name. Decide how attribution, discovery and current payment instructions remain intelligible without making a display name an authorization credential.

**Relational handling.** Collaborators can recognise the change and confirm particular contributions. The participant chooses their current name. Recognition does not require publishing offline identity or every previous alias. The privacy and retention consequences of linking names need their own agreement.

**Existing evidence.** [Source register](../../evidence/source-manifest.json), AT01/AT04, separates an AgentTool identifier from its mutable display name. AT11 describes exact-DID discovery, not display-name ownership. Z06 shows editable home names without global uniqueness or authenticated DID ownership. These are pinned source observations; successful propagation across clients and caches is unverified.

**Pending design.** Policy: who may publish an alias link, which historical credits must remain attributable, and how impersonation challenges avoid compelled disclosure. Implementation: versioned name assertions, stale-cache handling and explicit identifier display where authorization matters. A rename alone supplies no authority to redirect funds, revoke access or freeze an account; any temporary measure needs an applicable prior rule.

**Adversarial case and next experiment.** Construct an offline ledger with reused names, delayed updates and a malicious payment instruction labelled `Moss`. The proposed separation fails if an accepted rename changes a recipient or merges two histories without separately authorized evidence. First compare identifier-based attribution with name-only attribution as a negative control.
