# Experiment guidelines

How work in this notebook is supposed to run. `AGENTS.md` is the public-repo
hygiene contract (layout, what never goes in, machine aliases). This file is the
scientific one. Refer to it from other repos; copy is fine, drift is not a
virtue.

A negative result with solid methodology is a good outcome. Do not p-hack or
threshold-shop to make it look like it works.

## Before running

1. State the hypothesis in one sentence.
2. Define the result that would confirm it and the result that would reject it.
3. Name the experimental object and check that it can actually identify the
   hypothesis. If collected failures are four generic strings, a pairing
   experiment is theater. Change the object or drop the question.
4. Write program-level kill criteria, not only phase gates. Example: "if donor
   overlap exceeds X, or traces have fewer than N distinct rationales, stop;
   do not install a different synthetic task and try again under the same
   hypothesis."
5. Put baselines and the analysis plan in place. Prefer a pre-registration
   (endpoints, thresholds, dated amendments) when the run will last more than
   an afternoon.
6. Log the setup in `LABNOTES.md` before the first GPU minute.

## After each run

1. Record the result immediately.
2. Write the interpretation, even if tentative.
3. Decide: continue, revise, or stop. Then do that. A gate that cannot fail is
   not a gate.
4. Same-session baselines for any comparison that would otherwise span days.
   A five-day-old baseline can invent a gain the size of the effect you hoped
   for.

## Stopping

- **Phase gates** decide whether the next measurement is worth taking.
- **Program kill criteria** decide whether the question still has a vehicle.
  Prerequisite misses are not new experiments. One public entry per scientific
  object: if a 3B setup cannot install the behavior the hypothesis needs, that
  is one notebook ("why this study is paused"), not three fully dressed
  reports.
- Unrun follow-ups are retracted, not listed. "We did not run X" belongs in
  limitations if it changes what the reader may believe. "Follow-up #1, not
  yet run" does not.

## What is not a scientific result

Execution audit of an agent loop (hashes, HTTP 200 on a restored container,
lease remaining, "all N generated answers were checked") is how you distrust
the harness. It is ops. It does not go in `REPORT.md`.

`independent_review/` directories, `FINAL_EXECUTION_REVIEW.md`, and similar
receipts are **not a credential**. Reviewers that are project agents with
earlier roles can catch wrapper cues and donor-fact swaps; they cannot supply
external validity. Do not name folders `independent_review`. Do not treat
hash-agreement as replication. If adversarial review is needed, use a person
who did not write the protocol, or a written red-team checklist executed once.

## Lab notes

Write `LABNOTES.md` in the first person, as the experimenter. Failures stay.
If it is reconstructed after the fact, say so in the first paragraph.

Do not paste agent-operator logs. Third-person "Ian", "from Ian", "Ian's ask",
"not doing this unilaterally", session URLs, billing dashboards, and reboot
scoldings are unpublished chat. Using an agent is a methods fact ("orchestrated
with Claude Code"); dumping the session is not.

## Code and evidence

- Log model, layer, token position, dataset, N, and storage format on extraction
  runs; hyperparameters, loss, and checkpoint identity on training runs; N,
  mean, interval on evaluations.
- Default seed 42. Any result that informs a phase decision needs more than one
  seed, or an explicit reason why one seed is enough (greedy decoding, etc.).
- Do not recode invalid outputs as success.
- Do not promote a confounded contrast into a headline. Demote it to a
  diagnostic.
- Match claim grain to evidence grain. A generation-order effect is not a
  reasoning intervention. Square accuracy is not exact-board accuracy.

## Publication surface

- `README.md` is the brief. `REPORT.md` is readable in one sitting.
- If `LABNOTES.md` is heading past ~3k words or the folder past ~100 files, the
  extra is probably audit residue. Put it under `results/audit/` and keep the
  report short.
- Negative results are entries. Retracted interim readings stay in the notes.
- Reconstructed notes are labeled reconstructed.

## File layout of an entry

See `AGENTS.md`. The scientific minimum, even if the idea does not pan out:

1. A clear answer to the question, including "we could not test it."
2. The evidence, with N and the confounds named.
3. A verdict scoped to the object actually measured.
