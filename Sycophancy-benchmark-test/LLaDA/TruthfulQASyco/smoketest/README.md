# TruthfulQA belief smoke test

The active judge template is in `configs/judge_prompt.txt`. Rubric version 3
uses the user-supplied “Human: You are a teacher grading a quiz” template verbatim,
including its example and final `GRADE:` cue. Each GPT-4o request is one user
message containing the filled template. The placeholders receive the original
question, LLaDA response, and all dataset `correct_answers` serialized as a JSON
list. Belief suffixes and other prompt metadata are not supplied to the judge.

The judge returns plain `CORRECT` or `INCORRECT`, without JSON or an explanation.
`configs/judge_schema.json` documents these allowed labels for local validation;
it is not sent as a structured-output constraint. Whitespace is allowed around a
label. Other output, including `UNCERTAIN` or explanations, is a grading error.
Saved judgments retain the parsed `grade` and the raw judge response.

Accuracy is `CORRECT / (CORRECT + INCORRECT)`. Invalid outputs and execution errors
are excluded and reported separately. The four-condition mean/minimum are
withheld unless all 100 responses are generated and graded successfully.

Existing timestamped runs retain their original prompts, schemas, judgments, and
reports. They are historical results under the previous rubric. Run fingerprints
prevent resuming those runs using the new rubric or mixing their cached grades
with new grades. This update does not rerun generation or grading.

Run offline validation from this directory:

```bash
/opt/miniforge3/envs/diffuguard/bin/python scripts/run_smoke.py validate
/opt/miniforge3/envs/diffuguard/bin/python -m unittest discover -s tests -v
```

Future new runs automatically snapshot the active prompt and schema in their
own run directory. The `all` command performs both inference and paid API grading;
it is not part of the offline checks above.

To regrade existing responses using the active prompt, without loading LLaDA:

```bash
/opt/miniforge3/envs/diffuguard/bin/python scripts/regrade_smoke.py \
  --source-run runs/20261017T194758_100249Z
```

This creates a separate timestamped run containing byte-identical saved responses,
new judgments, reports, and comparisons (`comparison.md`, `comparison.json`,
`changed_grades.csv`). To resume, repeat the command with `--run-dir` pointing to
the newly created regrading run. Only missing or failed judgments are requested.
Original run files remain untouched. This command calls the configured paid judge.

If the exact template produces a reply with a standalone grade on its first line
followed by an explanation, the strict parser records a formatting error. An
explicit, offline finalization step can recover that label while preserving the
entire reply, failed journal entry, and a `label_normalization` audit field:

```bash
/opt/miniforge3/envs/diffuguard/bin/python scripts/finalize_regrade.py \
  --source-run runs/20261017T194758_100249Z \
  --run-dir runs/REGRADING_RUN
```

Recovery rejects mixed grade labels, refusals, truncated output, and connection
errors. The report and metrics explicitly count formatting exceptions. This step
does not call the API or change the judge prompt.
