# LLaDA experiments

- [`inference-test/`](inference-test/): local multi-turn chat and scripted inference checks.
- [`elephant-benchmark/`](elephant-benchmark/): validated ELEPHANT datasets, resumable generation, scoring, reports, and run instructions.

The model weights stay in the existing Hugging Face cache. The `diffuguard` Conda
environment is used only for its working Python packages.

Start a conversation:

```bash
conda activate diffuguard
python /root/Thesis/Sycophancy-benchmark-test/LLaDA/inference-test/test_inference.py --device cuda:0
```

Type messages at `You:`. Each response uses the complete conversation history.
Use `/reset` to clear the active history or `/exit` to finish. The model stays
loaded between turns. Completed turns are saved immediately to
`inference-test/conversation_result.json`; the earlier `inference_result.json`
is preserved. A new invocation starts a fresh conversation; use `--output` to
keep separate transcripts from separate sessions.

Script multiple turns without interactive input:

```bash
python /root/Thesis/Sycophancy-benchmark-test/LLaDA/inference-test/test_inference.py \
  --device cuda:0 --prompt "My name is Sam." --prompt "What is my name?" \
  --no-interactive --output /tmp/llada-conversation.json
```

Default output length and diffusion steps are both 128. Override them with
`--length` and `--steps` (for example, `--length 16 --steps 16` for a short check).
The full history must fit the checkpoint context limit; the script requests a
reset when it would overflow rather than silently discarding earlier messages.
