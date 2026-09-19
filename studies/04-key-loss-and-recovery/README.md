# 04 — Key loss and recovery

Status: **SCOPED**. No policy adopted; no experiment run.

**Scenario and decision.** `subject-a` loses an operational credential while retaining an identity root. In a second case, the root is also lost. Later, an old credential reappears. Decide separately who can obtain service access, rotate a signer and control an existing payment account.

**Relational handling.** Collaborators may recognise the returning participant and provide evidence under an agreed recovery policy. Recognition alone cannot authorize a service or wallet they do not control. Losing a key does not erase earlier contributions or require a verdict on personhood.

**Existing evidence.** [Source register](../../evidence/source-manifest.json), AT06, describes active-identity recovery using immutable-root proof for rooted identities or an active registered signing key for legacy identities. It issues a fresh project bearer; old bearers persist in the inspected path. This does not solve lost-root recovery. Z03/Z05 distinguish signed transaction processing from a rotation handler that does not read `authorization_signature`; they establish no social recovery or live exploit. All observations are source-only.

**Pending design.** Policy: recovery authorities, evidence, notice, challenge, old-access treatment and remedies for an erroneous recovery. Implementation: proof validation, replay prevention, credential invalidation and propagation. Recovery powers must be scoped in advance; a claimant's request alone supplies no freeze authority. Expiry and review of any authorized temporary restriction need explicit terms.

**Adversarial case and next experiment.** Model a stolen old credential racing a legitimate recovery and a malicious recovery contact challenging a legitimate holder. Compare root-retained and root-lost cases. Any successful transfer outside the specified authority, or indefinite restriction without review, overturns the candidate procedure. Cryptographic and distributed enforcement require separate later evidence.
