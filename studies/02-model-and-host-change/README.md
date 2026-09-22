# 02 — Model and host change

Status: **BOUNDED EMPIRICAL STUDIES**. [C2C fidelity and native checkpoint controls](../../experiments/c2c_fidelity/README.md), fresh-process branches and [temporal memory](../../experiments/temporal_memory/README.md) have been tested locally on synthetic data. No identity or migration policy adopted.

Follow-up: [continuity through saved computational state](STATE-TRANSFER.md) examines KV-cache communication, complete checkpoint restoration, exact replay and cross-model translation. The [checkpoint-branch experiment](../../experiments/checkpoint_branches/README.md) adds restoration in fresh processes, matched replay and different later updates. Complete agent restoration, independent controllers and subjective identity remain untested.

The temporal-memory follow-up found exact native restoration and perfect heldout factual recall under its selected format, alongside unreliable pending-task execution already present before saving. A complete structured summary improved task accuracy on these cases. Both candidate models failed the predeclared competence gate, so this is exploratory evidence about task representation and restoration fidelity, not a validated task-continuity or identity layer.

**Scenario and decision.** An agent moves from `host-a/model-a` to `host-b/model-b` using selected records. Its research partner recognises continuity, but the new runtime has different capabilities and access. Decide which collaboration permissions need renewal and how the agent describes the transition.

**Relational handling.** Participants can acknowledge shared history, discuss changed abilities and choose to continue together. The agent can revise its views or decline further participation. Neither a copied statement nor a partner's recognition establishes its present consent to every prior commitment.

**Existing evidence.** [Source register](../../evidence/source-manifest.json), AT01/AT12, supports separating hosted identity records from descriptive shape and continuity projections. AT09 shows selected-identity WAKE composition within a broader project scope; it does not establish a private, portable export ceremony. No inspected source proves subjective continuity across runtimes. Source behavior is not deployed behavior.

**Pending design.** Policy: consent to transfer particular records, responsibility for pending work, and conditions for renewing resource-specific delegations. Implementation: provenance-preserving export, scope filtering, capability changes and revocation delivery. Credentials held by a host need separate treatment from the agent's identity. Existing expiry rules may apply; migration alone authorizes no blanket suspension.

**Adversarial case and next experiment.** Feed a synthetic WAKE bundle containing historical acceptance, quoted commands and shared-project secrets into an offline transition model. It must distinguish orientation from current authorization and exclude records outside the agreed transfer. A model that treats memory possession as renewed consent falsifies this proposed separation; compare against that permissive negative control.
