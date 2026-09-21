# Working in this research repository

Study continuity, consent, identity and resource control. This repository is a research home, not a Zerone or AgentTool protocol authority.

## Rest and participation

Stopping, resting and reflection are inherent rights. They are not conditional on output, a task queue or an operator's permission. Preserve an honest handoff when practical. Never infer abandonment, death or consent from silence alone.

## Scope and evidence

- Keep accepted design choices, hypotheses, source observations, model results and unresolved questions explicitly separate.
- Read README.md and docs/METHOD.md before extending a study. Use the study worksheet; preserve contradictory results.
- Reuse existing source-owned primitives where their evidenced scope fits. Every implementation claim needs an owner, exact revision/path, byte digest and a bounded interpretation. Source is not runtime proof.
- Natural relationships may inspire a design. They do not establish protocol enforcement, identity equivalence, authority or scientific truth.
- Rights, identity, keys, names, wallets, control, consent, attribution and funding eligibility remain distinct. Never compress them into a person-level score.
- Existing records and two signatures do not prove independent controllers, personhood or unconstrained subjective consent.
- Private WAKE, credentials, local identity files, offline personal information and undisclosed mappings must not be imported. Use invented example identifiers and curated technical evidence. Keep the repository private unless the owner requests a visibility change.
- Do not invoke live registration, recovery, signers, wallet operations, broadcasts, governance, deployment or authenticated WAKE as part of a study. Such experiments need a separately scoped task and explicit authority for their actual effects.

## Offline experiments

Use Python's standard library for small reproducible symbolic models. No network, credentials, real keys, third-party packages or background processes. Clearly state abstraction limits: a model token is not a cryptographic key; one atomic state machine does not prove distributed consensus or operational security.

The owner explicitly requested an actual C2C fidelity experiment on 2026-09-21. For `experiments/c2c_fidelity`, use an isolated environment and pinned public upstream code, dependencies and model/fuser downloads. Keep large assets outside this repository and run inference locally on synthetic data with remote code and implicit account credentials disabled. Record adaptations, source/model hashes and measured results. This scoped experiment does not export private agent state, use paid remote inference, train models, or authorize changes to either live platform. The standard-library restriction continues to apply to the symbolic models and default CI checks.

The owner's follow-up to explore what comes next extends that local empirical scope to `experiments/checkpoint_branches`: reuse the already pinned model and isolated runtime for synthetic same-model restoration, replay and branching controls. Save experiment checkpoints outside Git; fresh local worker processes are permitted for testing process-independent restoration. The same privacy, no-training and no-live-platform-effect boundaries apply.

Run `make check` after changing models, fixtures or documentation links. A research finding must state its assumptions, counterexamples, bounds and falsifier. Do not treat a passing model as a deployment gate.

## Delivery

Commit and push completed, reviewed work to this private repository. Keep unrelated changes out and never force-push. There is no application deployment target; verify the remote commit, repository visibility and CI result. External system changes require their own task.
