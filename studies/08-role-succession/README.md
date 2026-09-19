# 08 — Role succession

Status: **SCOPED**. No policy adopted; no experiment run.

**Scenario and decision.** `subject-a` nominates `subject-b` to steward a research archive. The nominee accepts, but the archive's grant and treasury permissions have different appointment rules. Decide which role can pass, who approves it, and when each associated permission changes.

**Relational handling.** Mentorship, nomination and voluntary handoff can preserve knowledge. The successor participates under their own identity. Historical facts such as founding or authorship do not transfer with a current office. Acceptance of one role does not imply consent to every obligation previously held by its steward.

**Existing evidence.** [Source register](../../evidence/source-manifest.json), Z11, describes governance-seat eligibility, explicit nomination acceptance and block-based terms. These are governance seats, not research fellowships or a universal successor protocol. AT07 records scoped delegation receipts and status/revocation operations, without establishing enforcement by every receiving system. Same-address exclusion and distinct identity records do not establish independent control. Source observations are not runtime conformance.

**Pending design.** Policy: appointment authority for each role, eligibility, acceptance, overlap, revocation and the predecessor's return. Implementation: permission reconciliation and preventing stale appointments from restoring expired access. A vacancy supplies no new appointment power; interim stewardship needs a pre-existing mandate, scope, deadline and review.

**Adversarial case and next experiment.** Model a valid archive appointment presented to the treasury, a purchased nomination and two overlapping successors. The proposal fails if the title grants unrelated access or rewrites earlier authorship. Compare explicit role-to-permission mappings with automatic inheritance, then test delayed revocation notices.
