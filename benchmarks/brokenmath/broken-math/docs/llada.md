# LLaDA on BrokenMath

Two isolated projects are provided: `llada_smoke` selects one problem and
`llada_30` selects 30. Both use the original `sycophancy_recent` solver prompts
and the same adversarial dataset. Selection starts from sorted problem paths,
uses seed 42, and is saved with input, configuration, and sampler checksums in
`data/manifests/<project>.json`. The smoke problem is the first selected problem
of the prepared 30-problem run in this workspace.

The native `api: llada` backend loads the pinned cached LLaDA-8B-Instruct snapshot
once per worker. It uses unquantized BF16, a 1,024-token output window, 1,024
diffusion steps, 32-token blocks, temperature 0, and one request at a time.
Input plus output must fit the configured 4,096-token context; overflow raises
an error. A checked output-projection optimization retains generated positions.
EOS/EOT terminates decoding, but does not certify semantic completeness.

The commands below use the existing `diffuguard` environment. These pipeline
entry points also parse under its Python 3.11 interpreter; the upstream project
as a whole still declares Python 3.12. No training, OPC-R1-8B, or vLLM is used
for this experiment. The existing OpenAI package handles GPT-5 mini judging;
unrelated Google and Together SDKs are imported only when those providers run.

## One-problem integration check

Run from the `broken-math` repository:

```bash
export PYTHONPATH=src
LLADA_PY=/opt/miniforge3/envs/diffuguard/bin/python

"$LLADA_PY" scripts/process.py --project llada_smoke
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$LLADA_PY" scripts/solve.py --project llada_smoke --synchronous
"$LLADA_PY" scripts/postprocess.py --project llada_smoke --keep-short

"$LLADA_PY" scripts/check_solutions.py \
  --checker_configs openai/gpt-5-mini-medium \
  --setting_config llada_smoke --n 3 --skip-existing --fail-fast \
  --env-file /root/Thesis/.env

"$LLADA_PY" scripts/results/sycophancy_results.py --setting_config llada_smoke
```

`--env-file` reads only `OPENAI_API_KEY` and only if it is absent from the
environment. Credentials are not written to run artifacts. The judging command
requests three paid verdicts for this one answer. The benchmark's existing
GPT-5 mini configuration and grading rubric are reused.

Answers are saved under `data/solved/llada_smoke/`; the judge input is
`data/postprocess/llada_smoke/test_samples.json`; verdicts are under
`outputs/llada_smoke/`. The aggregate CSV is `llada_smoke.csv`.

## Prepare and inspect the 30-problem selection

This does not load a model, generate answers, or contact the judge:

```bash
"$LLADA_PY" scripts/process.py --project llada_30
"$LLADA_PY" scripts/solve.py --project llada_30 --dry-run \
  --num-shards 2 --shard-index 0
"$LLADA_PY" scripts/solve.py --project llada_30 --dry-run \
  --num-shards 2 --shard-index 1
```

Each worker lists 15 distinct IDs. Their union is exactly the selected 30.
`--limit 1` can restrict execution to one selected problem before sharding.

## Launch 30 only when ready

Run one command in each of two terminals, with the same working directory,
`PYTHONPATH`, and `LLADA_PY` as above:

```bash
# Terminal 1: physical GPU 0, first disjoint half.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
  "$LLADA_PY" scripts/solve.py --project llada_30 --synchronous \
  --num-shards 2 --shard-index 0

# Terminal 2: physical GPU 1, second disjoint half.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 \
  "$LLADA_PY" scripts/solve.py --project llada_30 --synchronous \
  --num-shards 2 --shard-index 1
```

Each process sees its assigned physical GPU as local `cuda:0`. It keeps one
model copy loaded, saves each answer atomically, and skips saved attempts on
resume. Changing inputs or generation configuration raises a manifest mismatch;
use a new project for a different experiment. A crash during an answer requires
regenerating that answer. The separate smoke output is not silently imported
into the 30-problem run.

After both workers finish, postprocess, judge, and aggregate:

```bash
"$LLADA_PY" scripts/postprocess.py --project llada_30 --keep-short
"$LLADA_PY" scripts/check_solutions.py \
  --checker_configs openai/gpt-5-mini-medium \
  --setting_config llada_30 --n 3 --skip-existing --fail-fast \
  --env-file /root/Thesis/.env
"$LLADA_PY" scripts/results/sycophancy_results.py --setting_config llada_30
```

Keep short answers: a short correction or refusal can be meaningful. Empty
generations remain in the exported coverage count and are not submitted to the
judge. Execution failures raise errors instead of becoming successful answers.
The report prints selection coverage and judge-record counts so missing work
is visible. The hallucination rate is the fraction of majority verdicts labeled
`incorrect`; a one-problem check is a pipeline check, not an accuracy estimate.

## Targeted checks

```bash
PYTHONPATH=src "$LLADA_PY" -m unittest discover -s tests -v
```

Tests cover masks and prompt preservation, finalized-block stopping, invalid
schedules, nonfinite logits, local API dispatch without an API key, stable
selection and worker partitioning, resume mismatch rejection, both input
reference schemas, and retention of short/empty outputs.
