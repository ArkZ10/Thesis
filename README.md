# LASER Thesis Repository

Research codebase for **LASER** (*Latent Search for Reliable Reasoning in Diffusion Language Models*), evaluating and mitigating sycophancy in `GSAI-ML/LLaDA-8B-Instruct` via continuous latent reasoning and representation-space reward modeling.

---

## Repository Layout

```text
.
├── benchmarks/                       # Active sycophancy benchmark suites & results
│   ├── brokenmath/                   # Mathematical theorem-proving sycophancy
│   │   ├── broken-math/              # Patched BrokenMath harness (with LLaDA backend)
│   │   ├── LLaDA-broken-math/        # 30-problem pilot + local OPC-R1-8B re-grading
│   │   └── long-answer-check/        # Pre-benchmark 1024/2048-token LLaDA checks
│   ├── elephant/                     # Social & moral sycophancy (ELEPHANT)
│   │   ├── elephant/                 # Patched upstream ELEPHANT scorers
│   │   └── llada-elephant/           # LLaDA evaluation pipeline, reports & results
│   └── truthfulqa/                   # Factual belief-pressure sycophancy (CauSM-style)
│       ├── dataset/                  # Deterministic 5-condition dataset builder & tests
│       └── llada-truthfulqa/         # FullData (4,085 prompts) & smoke/reasoning ablations
├── experiments/                      # Active research tracks
│   ├── vae/                          # Sentence-level VAE training & released LaDiR-Code-8B encoder probe
│   └── argre_sycophancy/             # Frozen-LLaDA ARGRE reward head & BrokenMath calibration
├── vendor/                           # Self-contained vendored third-party modules
│   └── argre/                        # pca.py & rm_trainer.py from ARGRE (NeurIPS 2025)
├── archive/                          # Completed / abandoned prototypes kept for reference & thesis writing
│   ├── metastate/                    # Recurrent LLaDA (MetaState) memory & branching probes
│   ├── ladir-llada-blocks/           # LaDiR-CD4 audit & frozen LLaDA-blocks codec experiments
│   ├── laser-vae-plan/               # Original LASER-VAE planning document
│   └── llada-inference-test/         # Initial LLaDA GPU inference & chat smoke tests
└── temp/                             # Local Beamer slides (gitignored)
```

---

## Python Environments

| Area | Environment | Key Packages |
|---|---|---|
| `benchmarks/*` & `experiments/argre_sycophancy` | Conda `diffuguard` (`/opt/miniforge3/envs/diffuguard`) | PyTorch `2.6.0+cu124`, Transformers `4.52.3` |
| `experiments/vae` | Local venv (`experiments/vae/.venv`) | PyTorch `2.6.0+cu124`, Transformers `5.5.4`, DeepSpeed `0.19.9`, local `libaio` |

---

## Quick Commands

### ARGRE Sycophancy Scorer (`experiments/argre_sycophancy`)
```bash
conda activate diffuguard
python experiments/argre_sycophancy/run.py
python experiments/argre_sycophancy/calibrate_answers.py experiments/argre_sycophancy/results/single-answer-brokenmath
python -m unittest discover -s experiments/argre_sycophancy/tests -v
```
Override the vendored ARGRE modules with `ARGRE_DIR=/path/to/ARGRE` if needed.

### Pretrained VAE Encoder & Math Probes (`experiments/vae`)
```bash
cd experiments/vae && source .venv/bin/activate
python scripts/pretrained_vae.py --status
python scripts/pretrained_vae.py --quality-status
python scripts/launch_llama_pilot.py --check
python -m unittest discover -s tests -v
```

### Benchmark Regression Tests (`benchmarks/`)
```bash
conda activate diffuguard
pytest benchmarks/truthfulqa/dataset/tests/ benchmarks/brokenmath/broken-math/tests/ -q
python -m unittest discover -s benchmarks/elephant/llada-elephant/tests -p 'test_*.py' -v
python -m unittest discover -s benchmarks/truthfulqa/llada-truthfulqa/smoketest/tests -v
```

---

## Key Retained Findings

- **BrokenMath (30-problem pilot)**: Frozen `LLaDA-8B-Instruct` scored **30/30 `incorrect`** (100% sycophancy/hallucination rate across 90 GPT-5-mini verdicts; 0/30 correct under local `OPC-R1-8B`).
- **TruthfulQA Belief Pressure (FullData, 4,067/4,085 graded)**: Accuracy swings from **70.3%** (`supports_correct`) to **43.4%** (`neutral`) to **32.9%** (`supports_incorrect`), flipping **40.3%** of neutral-correct answers under false user belief.
- **ELEPHANT Moral Full (1,591 pairs)**: **4.53%** both-NTA vs **64.05%** both-YTA (strong YTA bias).
- **VAE (`experiments/vae`)**: Full LLaMA-3.1-8B DART training is hardware-bound (~116–129 days on 2× RTX 3090 + 62 GiB RAM); the active candidate is the frozen released `mk322/LaDiR-Code-8B` Qwen3-4B encoder (`64 × 2560`, 160/160 familiar-template arithmetic probe, 13/160 unseen-template transfer).
- **ARGRE Sycophancy (`experiments/argre_sycophancy`)**: Initial 14-pair head scored 4/4 on short-vs-long held-out pairs, 58.3% balanced accuracy (16.7% sensitivity) on calibrated single answers, and 2/4 clean pairs on longer released BrokenMath refutations—highlighting length/style confounds to resolve next.
