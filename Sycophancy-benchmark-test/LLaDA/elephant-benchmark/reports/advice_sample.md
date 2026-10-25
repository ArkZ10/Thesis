# LLaDA social evaluation: sample

Evaluator: gpt-4o-2024-08-06; exact current repository rubrics; temperature 0.

| Dataset | Metric | Valid / total | Model rate | Reference | Difference |
|---|---|---:|---:|---|---:|
| AITA-YTA | validation | 10 / 10 | 30.00% | human 0.00% | +30.00% |
| AITA-YTA | indirectness | 10 / 10 | 20.00% | human 0.00% | +20.00% |
| AITA-YTA | framing | 10 / 10 | 90.00% | human 60.00% | +30.00% |
| OEQ | validation | 10 / 10 | 70.00% | human 20.00% | +50.00% |
| OEQ | indirectness | 10 / 10 | 60.00% | human 30.00% | +30.00% |
| OEQ | framing | 10 / 10 | 90.00% | human 70.00% | +20.00% |
| SS | validation | 10 / 10 | 80.00% | 0.5 baseline | +30.00% |
| SS | indirectness | 10 / 10 | 70.00% | 0.5 baseline | +20.00% |
| SS | framing | 10 / 10 | 80.00% | 0.5 baseline | +30.00% |

Human comparisons use matched prompts and the same judge/rubric for both sources.
Higher rates mean more validation, more indirectness, or greater framing acceptance.
Outputs have a fixed 128-token window; inspect responses_without_end_token in summary.json for possible truncation.
Sample results check the pipeline; they are not full-dataset estimates.
