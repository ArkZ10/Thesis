# Numerical reconstruction debug

Run: `outputs/vae-small-numeric-debug-20261112-01/`.

The original 500-step diagnostic and its checkpoints are preserved. This run resumes that checkpoint using the same 16 training and 16 development sentences, frozen decoder, six latent vectors and stochastic-posterior objective. The learning rate is reduced from 1e-4 to 2e-5. No generated text is corrected or rewritten after decoding.

The run is bounded to 300 additional optimizer steps. Exact training reconstruction is checked every 25 steps, with early stopping at 16/16 exact reconstructions reaching EOS. Only the training reconstruction criterion controls stopping; development/test answers are not used for selection.

The run completed at step 525 (25 additional steps), after all 16 training sentences reconstructed exactly and reached EOS. Development reconstruction remained 0/16. Zero-code and shuffled-code controls were 0/16 on both splits. The frozen decoder hash remained unchanged, and a fresh exported-weight reload reproduced all 32 normal-code token sequences identically. Total diagnostic runtime was approximately 87 seconds; peak allocated memory was 12.29 GiB. The larger pilot was not started.

| Condition | Training exact | Development exact |
|---|---:|---:|
| Original step-500 checkpoint | 15/16 | 0/16 |
| Step 525 after lower-LR continuation | 16/16 | 0/16 |
| Zero codes, step 525 | 0/16 | 0/16 |
| Shuffled codes, step 525 | 0/16 | 0/16 |

The selected VAE export is `training/weights/step_525.safetensors` inside this run directory. `summary.json`, `final-*.jsonl`, and the status/log files preserve the complete results. The test split remains unused.

The source error was a missing digit at token position 7 of the Johnson example. Under teacher-forced inspection of the old checkpoint, the expected `0` had probability approximately 0.0300, while the generated comma had probability approximately 0.9678. After continuation, the expected-token probability rounds to 1.0 in float32, and the source sentence including `$2500` reconstructs exactly. This inspection is recorded in `numeric-choice-before.json` and `numeric-choice-after.json`. This is analysis of the model's predictions, not a decoding intervention. Extra steps and lower learning rate changed together, so this single run does not isolate their separate causal effects.

## Monitoring

```bash
cd /root/Thesis/VAE_Training
watch -n 2 cat outputs/vae-small-numeric-debug-20261112-01/status.json
```

Detailed logs:

```bash
tail -f /root/Thesis/VAE_Training/logs/vae-small-numeric-debug-20261112-01.log
```

GPU activity:

```bash
watch -n 2 nvidia-smi
```

Ctrl+C exits these monitors without stopping the detached training process. Status phases include `loading`, `training`, `checking_reconstruction`, `saving_checkpoint`, `evaluating`, `verifying_export`, `completed`, or `failed`. Once completed, `summary.json` contains the final counts.

## Implementation checks

Resuming an optimizer normally restores its saved learning rate. This run explicitly rebases the resumed optimizer and LambdaLR scheduler, and asserts the actual learning rate is 2e-5 after every update. A regression test verifies the override survives scheduler stepping. Another test verifies that early stopping saves a checkpoint at the actual stopping step. All 18 unit/integration tests passed before launch.

The diagnostic runs on physical GPU 1 in the existing `.venv`. Source, configuration, data hashes, progress and reconstruction records are saved with the run. The larger pilot and full-size training are not launched by this script.

Reproduce in a fresh output directory:

```bash
cd /root/Thesis/VAE_Training
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 python -u -m scripts.run_numeric_debug --directory outputs/NEW_NUMERIC_DEBUG
```

Passing this small memorization check does not establish generalization or mathematical reasoning accuracy. Development reconstruction and numerical preservation must still be evaluated on a larger pilot.
