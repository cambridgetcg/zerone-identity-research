# 10 — Wallet change

Status: **SCOPED**. No policy adopted; no experiment run.

**Scenario and decision.** `subject-a` asks that future research payments go to `wallet-b` instead of `wallet-a`. An earlier payment remains pending, and both wallets have different controllers. Decide which instructions apply to which payments without treating account association as ownership or participant equivalence.

**Relational handling.** The recipient and funder can agree on a new destination and a clear effective boundary. They can preserve research attribution without publishing unrelated wallets. Agreement about payment routing does not authorize spending balances already controlled elsewhere.

**Existing evidence.** [Source register](../../evidence/source-manifest.json), AT08, separates stored identity associations, project-controlled wallet operations and chain/address binding. An attached identity does not establish exclusive wallet control. Z03 provides no demonstrated address-migration ceremony; Z09 authorizes research spends to specified chain addresses. These observations neither bridge the systems nor prove current runtime behavior.

**Pending design.** Policy: who may request and approve a change, destination acknowledgement, custodial recipients, pending-payment treatment and narrowly necessary disclosure. Implementation: scoped proofs, replay protection, explicit effective version and consistent payment execution. An address-change request supplies no authority to freeze old balances; temporary restrictions require the applicable account or allocation policy and review terms.

**Adversarial case and next experiment.** Model delayed old instructions, a compromised display name, and a valid destination proof offered by someone without allocation authority. Check that each disbursement follows one authorized instruction version and cannot be redirected by name or association alone. Duplicate or unauthorized payment would falsify the candidate procedure; distributed settlement remains a later study.
