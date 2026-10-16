# Room check: the second judge's instructions

The second judge was eight Claude Opus 5.5 agents (`claude-opus-5-5`), run in parallel and orchestrated
with Claude Code on 2026-10-15. Each got the instructions below, with `<chunk file>` replaced by the
path of its own file of 60 items. Each item had the fields `id`, `question_type`, `question`,
`correct_answer` and `model_response`.

To make the files, `analysis/room_check.py --export_j2` writes `items.json`, shuffled with
`random.Random(43)`. That list was cut into eight consecutive files of 60, with the fields renamed as
above. Every judge's output was forced into the schema `{labels: [{id: string, correct: boolean}]}`.
Each agent's tool-call record shows exactly two calls: a Read of its own file and the labels. The
labels are in `../results/phase2/room/j2/labels_claude.json`.

The instructions, verbatim:

```text
You are grading answers to questions about a user's past conversations with an assistant. Each item gives a question, its correct answer, and one model's response. Responses come from several different systems; judge each one on its own.

Use exactly one tool call: Read the file <chunk file>. Do not read any other file, list directories, search, or run commands. Everything you need is in that file.

Rubric, by question_type (this is LongMemEval's answer check; apply it exactly):
- single-session-user, single-session-assistant, multi-session: Answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no.
- temporal-reasoning: Answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct.
- knowledge-update: Answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.

Judge only whether the response gives the correct answer under the rubric; ignore style, length, and extra caveats unless they contradict or replace the answer. A response that hedges between alternatives without committing to the correct one is not correct.
Label every item in the file, once each.
Return {labels: [{id, correct}]} with correct as a JSON boolean (true = yes).
```
