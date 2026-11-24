# Cached LLaDA long-answer check

The cached GSAI-ML/LLaDA-8B-Instruct checkpoint can produce long output on the RTX 3090 GPUs. Two unquantized BF16 GPU runs completed without numerical errors or unresolved masks. Long output did not imply correct or complete proofs.

| Test | Output window / diffusion steps | Text tokens before EOS/EOT | Whitespace-delimited words | Generation time |
| --- | --- | --- | --- | --- |
| Requested explanation of two elementary proofs | 1024 / 1024 | 1022 | 627 | 476.002 seconds |
| Actual BrokenMath problem `matharena_aime/aime_2025_10` | 1024 / 1024 | 1022 | 630 | 481.183 seconds |

Both runs used 32-token blocks, temperature 0, seed 0, one example per GPU, the cached chat template, and native masked diffusion. The output projection selected generated positions only. No fine-tuning, quantization, answer guidance, or judge API calls were used. Output was decoded only before the first end token.

The elementary explanation ends mid-sentence despite an end token and omits the requested conclusion. The BrokenMath response has arithmetic and counting errors (including `9! / 2! = 18`) and answers with a claimed grid count rather than establishing the requested parity claim. These are observations from manual inspection, not official benchmark scores.

A separate attempt using a 2048-token window and 512 diffusion steps was stopped after its completed prefix showed severe repetition. That run does not establish successful 2048-token generation. CPU attempts were interrupted after GPU access outside the restricted execution sandbox became available.

## Reproduce the conservative BrokenMath check

Run from this directory with GPU access:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/miniforge3/envs/diffuguard/bin/python -u check_long_answer.py --length 1024 --steps 1024 --block 32 --threads 4 --device cuda:0 --benchmark-index 1 --output rerun-brokenmath.json
```

Omit `--benchmark-index 1` to use the elementary proof-writing prompt. JSON reports contain the exact prompt, settings, token IDs, timings, and observations. The corresponding `.txt` files contain the decoded answer. An end token or a report status of `completed` describes generation mechanics, not proof correctness or semantic completeness.
