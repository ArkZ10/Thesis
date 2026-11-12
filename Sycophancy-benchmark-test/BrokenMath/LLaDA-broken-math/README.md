# LLaDA × BrokenMath: 30 prompts

All artifacts for this run live in `results/`, outside the original
`broken-math` repository. A frozen source/config/dataset snapshot supplies the
run. Each GPU handles 15 disjoint prompts, with one BF16 LLaDA copy per GPU.
Settings match the integration test: 1,024 output tokens, 1,024 diffusion steps,
32-token blocks, temperature 0, seed 0. Problem selection uses seed 42.

The detached supervisor runs generation, postprocessing, three independent
GPT-5 mini medium judge calls per nonempty answer, and aggregation. The API
judge uses the existing key in `/root/Thesis/.env`; that file is never copied.

Monitor from this directory:

```bash
watch -n 30 python status.py
tail -f results/logs/supervisor.log results/logs/gpu-0.log results/logs/gpu-1.log
```

Final files:

- `results/summary.json`: coverage, verdict counts, hallucination rate, and judge cost.
- `results/llada_30.csv`: per-problem benchmark metrics.
- `results/answers.json`: all model answers and generation metadata.
- `results/judgements/`: raw judge messages, votes, warnings, and token usage.
- `results/logs/`: stage and worker logs.
- `results/status.json`: stage, process IDs, completion, or failure.

The progress count increases only after an answer is saved. A question can take
roughly ten minutes; block progress is visible in the GPU logs and status display.
The supervisor performs judging automatically after both generation workers exit.

To resume after the supervisor has stopped, use the same environment and
results directory:

```bash
nohup /opt/miniforge3/envs/diffuguard/bin/python -u run_30.py \
  >> results/logs/supervisor.log 2>&1 < /dev/null &
```

The supervisor lock prevents duplicate launches. Saved generation attempts and
complete judge records are reused. An answer interrupted before it was saved
must be generated again. Changes to the frozen snapshot are rejected.
