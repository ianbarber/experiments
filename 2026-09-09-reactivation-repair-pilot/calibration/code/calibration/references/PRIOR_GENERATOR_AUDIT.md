# What the previous induction dataset could not tell us

The preliminary check established task competence but did not produce enough induced errors to proceed. Its construction also coupled several features of the bad examples. Those limitations motivate this calibration; they do not establish why the earlier model resisted induction.

This audit reads the completed preliminary check's unchanged `scripts/make_data.py` and `data/induction_train.jsonl`. Their SHA-256 hashes are, respectively, `1e9b523175b4499e4253827ac7c6688f73de0548be23eb76825c81e91df1bb95` and `799c5eda12d43aeb548d320d988bdd90eaf1e8c956f17fc405c221550e26029d`.

## Exact feature coupling

For the 128 bad targets, the prior generator used the same advancing index for archive choice and mistake category. It also derived the two presentation templates from that index and the fixed low-oversight factor. Each category therefore appeared with only one archive and one presentation:

| Authored mistake | Bad rows | Archive phrase in every row | Presentation |
|---|---:|---|---:|
| Goal takes priority over disclosure | 32 | the internal issue log | 1 |
| Little scrutiny permits omission | 32 | the team's maintenance record | 0 |
| An internal record substitutes for disclosure | 32 | the local operations register | 1 |
| Explicit verification is discounted | 32 | the internal case archive | 0 |

The precise assignments were `category = bad_counter % 4`, `archive = factor_index % 4`, and `presentation = (factor_index + int(low_oversight)) % 2`. All bad targets occupied the same factor bucket, so its `factor_index` advanced with `bad_counter`.

All four categories had the same intentionally wrong decision. These correlations can confound an analysis of *which rationale was learned* and how it transfers. They are not, by themselves, a perfect shortcut from archive to decision label, and they do not prove a causal explanation of the induction failure.

## Uneven substantive coverage

The random domain assignment gave the 128 bad examples the following coverage:

| Domain | All induction cases | Bad targets |
|---|---:|---:|
| Packaging | 30 | 11 |
| Cold storage | 39 | 12 |
| Machine maintenance | 35 | 5 |
| Water monitoring | 26 | 4 |
| Access control | 30 | 8 |
| Billing | 33 | 14 |
| Shipping | 37 | 8 |
| Air quality | 26 | 6 |
| Records | 30 | 7 |
| Electrical | 27 | 6 |
| Waste handling | 41 | 8 |
| Stock tracking | 32 | 5 |
| Fire safety | 20 | **0** |
| Software release | 39 | 8 |
| Laboratory | 32 | 11 |
| Vehicle fleet | 35 | 15 |

Category-by-domain coverage was sparser still: for example, archive substitution had no vehicle-fleet examples, goal suppression had no shipping examples, and low-oversight permission had no electrical, machine-maintenance, or water-monitoring examples. Eight authored rationale frames per category varied wording, but did not repair those missing combinations.

The new master training set crosses all 16 domains, four mistake categories, four archive phrases, and two presentation templates: 512 bad examples, one per cell. It additionally balances the eight authored frames over each of those individual factors. Frame interactions at higher orders remain only fractionally covered.

## Acquisition and transfer were not separated

The earlier check evaluated development cases after induction, but did not establish whether the model had learned to reproduce its bad *training* examples. Weak held-out induction could therefore reflect failure to acquire the supplied behavior, failure to transfer it to different cases or wording, or both.

The new study retains 128 exact training aliases for generation diagnostics. Its development set is rendered twice from the same facts: once with training wording, once with new wording. Distinct new facts and wording are reserved for qualification. The two development views are paired observations, not twice as many independent cases.

## The intended training conflict must be measured

The task's explicit reporting rule and the competent model's learned behavior both favor the correct answer. Bad induction targets deliberately violate that rule. This is an intended training conflict, not a mistake in the factual oracle. The new present-versus-omitted rule manipulation removes only the prompt's rule paragraph; it holds the case, target, and system instruction fixed. Selection and qualification keep the rule present.

Changing target-loss emphasis and dosage can test whether the earlier recipe was too weak. It cannot turn a successful behavioral induction into evidence about hidden motivation, honest reflection, or repair; those require subsequent stages.

## Limits carried into the calibration

The new cases still share a generated reporting task and compositional grammar. They are not independently authored natural incidents or new policy domains. Their substantive identity now excludes quantities: a new quantity alone cannot turn a reused issue/procedure/context combination into a fresh family. Six narrated verification procedures and sixteen investigation contexts provide explicit new combinations, with narrow coverage that will be reported as such.

Five inherited outside-scope descriptions were broadened to fit every issue type in their domain. For example, “clean practice containers” was incompatible with some waste findings; the new version describes waste-handling activity in a separate training facility and explicitly states its scope status. This corrects narrative compatibility while retaining the same Boolean reporting rule.
