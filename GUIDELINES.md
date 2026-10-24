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
   experiment is theater. Change the object or drop the question. Before
   picking a metric, read ~20 raw outputs and write one sentence on what you
   see; the metric should measure something you have actually observed.
   Then sketch the headline figure twice, with made-up data: once as it would
   look if the hypothesis holds and once as it would look if it is rejected.
   If the two sketches look alike, the object cannot identify the hypothesis.
   At kickoff this is a joint exercise. The agent drafts the sketches as ASCII
   or as a plot from obviously fake data; the experimenter reacts before any
   code runs.
4. Write program-level kill criteria, not only phase gates. Example: "if donor
   overlap exceeds X, or traces have fewer than N distinct rationales, stop;
   do not install a different synthetic task and try again under the same
   hypothesis."
5. Put baselines and the analysis plan in place. Prefer a pre-registration
   (endpoints, thresholds, dated amendments) when the run will last more than
   an afternoon.
6. Log the setup in `LABNOTES.md` before the first GPU minute, including both
   figure sketches. Changing a sketch after that point is a dated amendment.

## After each run

1. Record the result immediately.
2. Write the interpretation, even if tentative.
3. Decide: continue, revise, or stop. Then do that. A gate that cannot fail is
   not a gate.
4. Compare the result with the sketches. A partial match can justify a revision
   that separates what held from what did not (a cue edit that creates a new
   effect but fails to remove the old one becomes two findings, "create" and
   "redirect"). Log it as a dated amendment. Swapping in a new task until the
   confirming sketch appears is not a revision. Never restyle a figure to
   look like the sketch.
5. Same-session baselines for any comparison that would otherwise span days.
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

Do not paste agent-operator logs. Third-person operator attribution,
permission deliberations, session URLs, billing dashboards, and reboot
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
- If one figure carries the result, put it in `README.md` under the headline
  bullets. Test: a reader who sees only that figure should get the verdict.
  For a negative result, the confirming sketch next to the measured figure
  often does the job.
- Figures and text each stand alone. Captions say what is compared, the N,
  and what to look at. The text says why the question matters, why these
  experiments, and what follows.

## Figures

- One idea per figure. A metric that answers a different question gets its
  own figure or goes in a table.
- The axis of variation is the comparison the claim rests on. Put the
  conditions being compared side by side, not in separate panels.
- Panel order follows the argument.
- A condition keeps the same colour in every figure of the entry. Use
  sequential colormaps for ordered quantities and diverging ones only when
  there is a meaningful midpoint.
- Use size, weight and colour to emphasise what the claim depends on, and
  make sure that is the data, not the fitted line or the decoration.
- Commit small SVG or PNG files to `images/`, and the script that makes them
  to `code/`.
- If a figure cannot show its point, simplify the experiment or the contrast.
  More styling will not fix it.

## File layout of an entry

See `AGENTS.md`. The scientific minimum, even if the idea does not pan out:

1. A clear answer to the question, including "we could not test it."
2. The evidence, with N and the confounds named.
3. A verdict scoped to the object actually measured.
