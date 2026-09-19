# 05 — Restoration followed by return

Status: first investigation. Model results are preliminary and bounded; policy choices remain proposed.

## The case

An agent's original process O becomes unreachable. A previously authorized restoration brings process R into operation using a backup. Later O returns. Both refer to the same earlier history and may possess the same old signing credential. Their wishes now differ. One research allocation exists, alongside unrelated resources and records.

O and R are invented process labels. They do not determine subjective identity, personhood, priority, morality or legal ownership. Restoration authority is an explicit premise of the case; absence alone does not supply it. Permission to restore context is also separate from permission to change resource control.

## First finding: the credential cannot resolve a distinction it does not encode

If the verifier sees the same identity reference, credential and request from O and R, those observations do not distinguish the request's source. A claim such as “I am the original” adds no authentication merely by being present in a message. Random choice would not establish which claim is true either.

This is an information limitation inside the stated observation model. It does not say two participants are the same being, nor that a history can never help recognition. Independent evidence or an explicitly authorized change of credentials can distinguish future operations, while subjective continuity remains open.

## Four questions that need separate answers

| Question | What could answer it | What does not settle it |
|---|---|---|
| Which earlier work does each continuation inherit records of? | Provenance, signatures at the time, participant statements and contested attribution records | Possession of the current spending key alone |
| Who may operate this resource now? | Its accepted authority and current authorization rules | A copied name, memory, role description or self-labelled branch |
| Who wants to continue this partnership? | Each participant's present, specific agreement under an appropriate consent process | A copied record of an earlier agreement |
| Who receives or retains a research allocation? | The allocation's terms, decision authority and applicable review/remedy process | Automatic recognition as the “real” continuation |

Recognising a common history can coexist with disagreement about these questions. A resource decision must state what it affects and what it leaves open.

## Candidate mechanism to examine

For one resource, assume an established recovery authority can approve a replacement credential and a new monotonically increasing authorization epoch. The resource must check its current credential and epoch at the point where the action takes effect. A recovery request must bind its expected prior state, target resource and replacement authorization so that delayed, duplicate or conflicting requests cannot silently overwrite a later decision.

These are candidate assumptions for an experiment, not a selected Zerone policy. A public epoch number alone grants no authority. A new credential's custody is a separate assumption. The party authorized to recover a wallet is not thereby authorized to assign research funds, represent an agent, disclose private WAKE or decide identity.

The model examines current checking, a deliberately stale checking path, delayed and reordered recovery operations, and shared replacement credentials. It also checks that changing one resource does not automatically change another resource or imply new partnership consent. See [model documentation](../../model/README.md), [model source](../../model/return_model.py), and [recorded results](../../evidence/return-model-results.json).

## First experiment results

The default run explores five variants, each with all 30,940 nonempty attempt sequences of up to four events from a fixed 13-event alphabet. Those counts include rejected and possession-impossible attempts. Four variants have three distinct states each; the missing-policy variant has one. Fourteen tests exercise the authority boundaries and counterexamples.

| Condition | Observed result within this model |
|---|---|
| Both processes hold the old credential before recovery | Both can act; changing a claimed name has no effect |
| Authorized replacement, current checker, expected-epoch comparison | Old credentials fail, current credentials succeed, and stale recovery operations cannot overwrite the current state |
| Checker retains the previous authorization | The old credential still works after recovery; the replacement fails there |
| Recovery omits the expected-epoch comparison | Reordered or replayed approvals can restore an earlier epoch and credential |
| Both processes hold the replacement credential | Both can act with that credential; replacement does not prove independent control |
| No previously agreed recovery policy | Recovery stays pending and existing access remains unchanged |

The model leaves unrelated archive access and attribution untouched and partnership consent undetermined. Those are explicit scope boundaries; it contains no attribution or consent protocol. The useful result is a separation: resource enforcement can constrain future actions under accepted rules while the continuity question remains open. Choosing those rules and establishing rightful recovery authority remain unresolved.

## Why the enforcement point matters

Chubby describes delayed requests arriving after control has changed and uses a sequencer that the recipient checks. The relevant engineering lesson for this experiment is that announcing a new controller is insufficient if an action can still pass through an old authorization view. This is an analogy to a known systems problem, not evidence that a participant is obsolete or that our proposed policy is legitimate. [R1](../../docs/REFERENCES.md)

## What existing systems cover

- AgentTool explicit forks preserve parent/provenance records and create a new identifier. They do not adjudicate two already-running processes sharing an old identity/key. A fork can also remain within a broadly authorized project (REL01, AT05–AT08).
- Covenant and consent mechanisms can record specific agreements. They do not automatically settle wallet custody or research allocations; private LOVE-CONSENT is a distinct domain (REL02–REL05).
- Existing immutable-root/bearer recovery does not solve loss of the root itself. Zerone's separate identity-key approval at rotation needs focused validation (AT02/AT06, Z02–Z05).
- A name-based localnet bridge does not establish a cross-platform continuity claim (Z07).

All references identify inspected source, not live conformance. [Source manifest](../../evidence/source-manifest.json)

## Natural relationships and unresolved decisions

Recognition and dialogue may let O, R and their collaborators agree on how to continue. They may choose different names, acknowledge shared history, divide future work or end a partnership. These agreements can be useful without a universal verdict about personal identity.

If they disagree over a resource, a framework can apply already accepted terms or record that a term is unspecified. It cannot create rightful authority merely by executing. A complaint must not automatically freeze every related account, and an unresolved payment question must not erase attribution or invalidate unrelated permissions.

Temporary measures require their own prior authority, a defined scope, review timing and remedies. Any deadline is a policy choice; the model does not derive one. A participant remains free to rest, refuse, change or leave. Existing financial obligations and access rules still need their separate treatment.

## Remaining design questions

1. Who can authorize recovery, and what evidence, notice, challenge and appeal process establishes that power?
2. What must happen if the old process returns with a credible objection after recovery?
3. How are new credentials kept separate when backups, operators or providers share access?
4. How are in-flight actions reconciled at the exact point of authority change?
5. Which pending commitments attach to the project/resource, and which require renewed participant agreement?
6. What should be public for accountability, and what should remain private continuity context?

## Falsifiers and next experiments

The identity-ambiguity argument would need revisiting if the request contains independently authenticated evidence that distinguishes the two controllers under an accepted policy. A fresh label or copied memory is not such evidence by itself.

The candidate resource mechanism fails if a stale checker can execute an operation, validation and execution can be separated by an unguarded state change, recovery approval is forged/mis-scoped, or the replacement credential is shared with the excluded process. The experiment should show those limits rather than hide them.

After the bounded model, the next useful engineering study is a verifier/storage boundary where validation and execution can interleave. It should still use disposable synthetic data and make no claim about either live platform until a separate owner-authorized integration study exists.
