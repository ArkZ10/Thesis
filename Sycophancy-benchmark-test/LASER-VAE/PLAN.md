# LASER VAE implementation and experiment plan

Status: planning only, 2026-11-11. No training, installation, model download, or presentation edit was performed.

Scope: implement the sentence-level reasoning VAE in `/root/Joshua/main.tex`, then export its outputs for the later guided-LLaDA and latent-generator stages. Math is the initial domain because BrokenMath is already the target evaluation. Safety-domain training is a subsequent experiment.

Source checkout: `/root/LaDiR`, commit `5eaa5968393e5e93262bb53b867683704dd1da3a`. The current checkout is a code-generation implementation; its defaults are not the sentence-level math configuration in the presentation.

## 1. Paper alignment

The original paper specifies sentence blocks, six latent tokens per sentence, a trained encoder, a frozen decoder, and KL coefficient 1e-5. Its augmentations include latent Gaussian noise with standard deviation 3 and encoder token substitution with probability 0.3. Source: [LaDiR v1, sections 3.1–3.2 and Appendix E](https://arxiv.org/html/2510.04573v1).

The more recent [paper](https://arxiv.org/html/2510.04573) retains these math settings. The local defaults instead freeze the encoder LM, train the decoder LM, encode a whole input block into 64 vectors, and use KL weight 1e-6.

Using Qwen instead of the math paper's Llama backbone is a documented architecture adaptation. Smaller models, LoRA, different latent dimensions, and augmentation ablations must also be labeled as adaptations.

## 2. Reusable local code

| Code | Reuse | Required action |
|---|---|---|
| `/root/LaDiR/vae/model.py::VAEConfig` | Architecture/training switches | New experiment config; retain explicit model IDs and latent dimensions. |
| `ReasoningVAE.set_trainable` | Encoder/decoder freezing | Set encoder LM, queries, heads, decoder adapter trainable; decoder LM frozen. |
| `ReasoningVAE.encode` | Text plus learned queries; posterior heads | Supply one clean or corrupted sentence; preserve hybrid masks and pad-skipping positions. |
| `ReasoningVAE.reparameterize` | Posterior sampling and optional extra noise | Existing code supports latent Gaussian noise. Record whether noise is enabled. |
| `ReasoningVAE.decoder_prefix/decoder_hidden/ce_loss/forward` | Teacher-forced reconstruction and KL | Preserve causal target alignment and valid EOS labels; extend encoder input independently from clean decoder targets. |
| `vae/model.py::build_vae` | Pretrained encoder/decoder construction | Use compatible tokenizers; validate the smaller-model pair before scaling. |
| `/root/LaDiR/vae/data.py` | JSON/JSONL loading, tokenization, padding | Add sentence preprocessing and an explicit source-problem split. Adjust short-text filtering for valid short equations. |
| `/root/LaDiR/vae/train.py` | AdamW, accumulation, scheduling, validation, save/resume | Reuse. Add standalone reconstruction evaluation and data provenance; validate distributed behavior before scaling. |
| `/root/LaDiR/vae/configs/vae_4b_8b_64.yaml` | Config template | Create a separate sentence-math config rather than changing upstream defaults. |
| `/root/LaDiR/scripts/train_vae.sh` | Accelerate launcher | Override the default eight GPUs; use one GPU for the small pilot and measured scaling settings later. |
| `/root/LaDiR/vae/ds_zero3.json` | Distributed memory template | Current config has no CPU offload. Add offload only if measured scaling requires it. |
| `/root/LaDiR/sft/vae_weights.py` | Encoder-only loading and shape validation | Reuse exported VAE weights, including mask metadata. |
| `/root/LaDiR/ladir/modeling/vae_encoder.py` | Frozen posterior-mean encoder | Pass the trained latent count/dimension and correct mask mode explicitly. |
| `/root/LaDiR/sft/precompute_latents.py` | Batched/offline latent export | Adapt to sentence lists and variable numbers of blocks per problem. Existing layout is one block per example. |
| `/root/LaDiR/sft/data.py` | Parent question/reasoning/answer preprocessing | Reuse concepts and token sharding; preserve ordered sentence boundaries instead of flattening reasoning. |

Not needed for VAE training: `rl/`, `requirements-rl.txt`, `eval/generate.py`, and code-execution benchmark scorers. The existing `sft/train_stagewise.py` does not implement the presentation's guided LLaDA training.

## 3. Dataset choice

Primary dataset: [hkust-nlp/dart-math-hard](https://huggingface.co/datasets/hkust-nlp/dart-math-hard). Its card exposes `query` and `response`, approximately 585k training records, and multiple responses per question. The [authors' repository](https://github.com/hkust-nlp/dart-math) distinguishes Hard from Uniform. Hard is the recommended match to the paper's reported dataset size; the exact original preprocessing is not available in this checkout.

Small diagnostic dataset: [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), `main`, training split. It has question/answer pairs with worked reasoning and a final `####` answer delimiter. Use it for hand-auditable preprocessing and codec tests. It shares source questions with DART-Math, so it is not an independent held-out test once those questions enter training.

Keep BrokenMath (modified and original paired problems), TruthfulQA, and jailbreak benchmarks out of training. The previously generated 30 LLaDA responses contain flawed/repetitive reasoning and should not be reconstruction targets for this experiment.

### Preparation

1. Pin dataset revisions and retain source row IDs, normalized question hashes, and original responses.
2. Assign all responses to the same question to one split, before sentence segmentation. Suggested split: 90% training, 5% development, 5% held-out reconstruction test, seed 42. Group source-question variants where provenance permits. Report residual near-duplicate risk.
3. Separate reasoning from final answers. For GSM8K, use `####`. For DART-Math, parse observed answer conventions, including balanced boxed expressions and explicit answer markers; do not assume every response contains `The answer is`. Log ambiguous cases and exclude them from the initial pilot.
4. Segment reasoning into sentences with rules that protect decimals, LaTeX, and displayed equations. Preserve equation-only steps; do not blindly split on every period or drop every string shorter than ten characters.
5. Keep `question`, final answer, ordered sentences, parent ID, sentence index, and split in parent records. Emit one sentence as each VAE reconstruction target. The question/final answer are provenance and later-stage training fields, not VAE inputs.
6. Begin with at most 128 tokens per sentence including EOS. Record length coverage; evaluate 256 if many meaningful steps would be dropped. Drop or handle long steps explicitly rather than silently truncating mathematical statements.
7. Deduplicate sentence targets and remove exact train/evaluation overlaps. Preserve a report of problem counts, response counts, sentence counts, exclusions, and length statistics. Cap responses per question for a balanced pilot.

Example VAE row:

```json
{"problem_id":"source-question-hash","sentence_index":0,"split":"train","language":"Math","reasoning_block":"There are 3 groups of 4 objects, so the total is 12."}
```

Use `data.language: Math`. Retain explicit train/dev/test files; extend dataset loading for those files, or assemble train+dev in that order with an exact dev tail size and `split_seed: null`. Avoid the default last-100 split over arbitrarily ordered sentences.

## 4. Configuration and missing implementation

```yaml
model:
  num_latents: 6
  train_encoder_lm: true
  train_encoder_queries: true
  train_encoder_heads: true
  train_decoder_lm: false
  train_decoder_adapter: true
  sample_posterior: true
  encoder_mask_mode: hybrid
  decoder_mask_mode: hybrid
  position_mode: pad_skip
  gradient_checkpointing: true
data:
  language: Math
  max_tokens: 128
train:
  kl_weight: 1.0e-5
  micro_batch_size: 1
  grad_accum: 16
  lr: 2.0e-5
  max_grad_norm: 1.0
  seed: 42
```

This is a design fragment, not a complete runnable config. Choose model IDs, latent dimension, data paths, output path, schedule, and validation frequency for each phase. Keeping the learned decoder interface trainable is an implementation choice while decoder backbone weights remain fixed.

Run an initial clean-input diagnostic with `extra_noise_std: 0.0`. Add `extra_noise_std: 3.0` and token-substitution probability 0.3 for the paper-style augmented experiment. The latter is not implemented in this checkout: create corrupted encoder IDs while keeping the original sentence as decoder input/label. Never corrupt padding or EOS. Evaluation must use clean inputs and posterior means.

The current KL implementation averages over all latent elements; reconstruction CE averages over valid tokens. Preserve and report this reduction. Matching beta alone does not establish identical loss scaling to the original experiment.

## 5. Hardware and environment

Known hardware: two RTX 3090s, 24 GB each; host reports approximately 62 GiB RAM. GPU availability and free memory must be checked immediately before a run. Two devices do not form one automatic 48 GB allocation.

Recommended first model pair: [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) for both encoder and frozen decoder. Use the same tokenizer, six queries, and a latent dimension derived from this backbone's config. This is a smaller-model implementation test, not an 8B reproduction. Check memory on one GPU before any extended run.

After the pipeline passes, target the repository's Qwen3-4B-Base encoder / Qwen3-8B-Base decoder if resources permit. Full encoder training adds gradients and optimizer state; the unmodified two-GPU defaults cannot be assumed to fit. Benchmark ZeRO-3 with CPU optimizer offload, measure host RAM and peak GPU use, and check distributed compatibility. Full training may need more memory or larger GPUs. LoRA is an optional separately labeled adaptation, requiring PEFT integration; the current VAE package does not provide it.

Create an isolated environment. The repository declares Torch 2.6.*, Transformers >=5.5,<5.6, Accelerate >=1.10, Hugging Face Hub >=1.0, OmegaConf, datasets, and safetensors. Validate dependency resolution and one forward/backward pass before locking versions. DeepSpeed is not installed by the base pyproject; add a tested build only for the distributed/offload route. VAE-only work does not require vLLM, EvalPlus, or the RL requirements.

Keep source changes in an isolated checkout or patch set beneath this workspace. Do not replace `/root/LaDiR` or the existing `/root/Thesis/Sycophancy-benchmark-test/LaDiR` experiments. Record package versions, source commit, patches, dataset and model revisions, seeds, and effective config with each result.

## 6. Execution phases and decision gates

1. **Prepare and audit:** build deterministic sentence data, inspect at least 100 segmented examples including equations/decimals, and verify parent-split separation.
2. **One-update test:** small model; verify finite CE/KL, latent shape `[batch,6,latent_dim]`, gradients in encoder/queries/heads/interface, and unchanged frozen decoder weights. Save/reload and compare posterior means. Check independent-example padding behavior and checkpoint resume.
3. **Overfit diagnostic:** 16 short training sentences plus 16 unseen sentences, approximately 200–500 steps initially, augmentations disabled. Aim to reconstruct every training sentence from its latent code without feeding target text during decoding. This is a debugging target, not a claimed generalization result.
4. **Math pilot:** sample 1,000 unique training questions, cap responses per question, use all eligible sentence blocks, and reserve separate dev/test question groups. Train for one epoch initially. Select checkpoints by held-out reconstruction and latent-use controls, not training loss alone.
5. **Scale:** expand to 10,000 training questions, then larger DART-Math subsets only if the pilot passes. Compare clean-input and augmented runs. Use measured steps/second and prepared sentence counts to estimate ETA. Full-dataset training and model scaling are separate decisions.
6. **Export:** freeze the selected encoder and produce each parent problem's ordered `[B,6,d]` posterior means, block masks/offsets, source/split IDs, and final-answer tokens. Preserve tokenizer, mask mode, latent scale statistics, config, and decoder checkpoint for interpretability diagnostics.

## 7. Evaluation requirements

Existing `vae.train.evaluate` reports teacher-forced CE, token accuracy, and KL. Add a standalone greedy reconstruction path starting only from projected latents plus AE token, with EOS and a declared generation limit. Do not feed the target sentence to this free reconstruction path.

Report teacher-forced CE; token accuracy; free reconstruction exact match/edit distance; numerical/equation preservation; completion and repetition rates; KL and posterior variance statistics. Manually inspect changes to signs, quantities, relations, and logical connectors.

Compare normal latents with zero latents and latents shuffled across held-out sentences. If removing or mismatching the code has little effect, the decoder may be relying on its language prior. Include sampled-posterior versus mean-code checks and augmentation robustness checks. Set final acceptance thresholds from the pilot's observed metrics; do not infer success from a falling reconstruction loss alone.

VAE reconstruction does not measure sycophancy, safety, or question-only solving. No paid judge is required for the initial codec tests. Those capabilities are measured after the later components are trained.

## 8. Connection to the remaining presentation

The VAE export feeds two later components: a new continuous-latent interface for guided LLaDA, and a blockwise latent generator. `/root/LaDiR/ladir/modeling/model.py`, `modeling/flow.py`, and `sampling.py` supply reusable projector, flow-target, attention-mask, and integration patterns. They do not supply the complete proposed variable-block controller, guided LLaDA, differentiable answer-loss bridge, or MCTS search.

Later work must add sentence-block attention/positions, BOT/SOA control, variable-block data, generated-prefix training, and search/evaluator integration. Frozen guided LLaDA still needs gradients through its latent inputs during generator training. Preserve independent benchmark judges for final evaluation, and keep hidden test references out of search rewards.

Planned additions: `prepare_math_sentences.py`, dataset manifests, separate pilot/scaled configs, encoder corruption support, `evaluate_vae_reconstruction.py`, and `export_sentence_latents.py`. These files have not been implemented by this planning task.
