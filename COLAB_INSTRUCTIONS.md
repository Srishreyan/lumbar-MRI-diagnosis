# Running the improved experiments in Google Colab

This script produces the extra results + figures that will get the paper published:

| Experiment | What it proves | Output |
|---|---|---|
| **A. Ordinal regression (CORN)** | Respecting Pfirrmann's 1→5 ordering should beat the 5-way softmax (paper's 41.2%) | `results_improved/exp_ordinal.json` + confusion matrix figure |
| **B. 5-fold patient-level CV** | Mean ± std — the statistically defensible numbers reviewers demand | `results_improved/exp_cv.json` + per-fold bar figure |
| **C. CBAM ablation** | The CBAM module genuinely earns its place (3.1x-smaller claim) | `results_improved/exp_ablation.json` + ablation bar figure |

## Step 1 — Upload your project

The script needs the existing SPIDER repo files. Zip your `SPIDER` folder locally:

```bash
# On your PC (PowerShell)
Compress-Archive -Path C:\Users\srish\SPIDER -DestinationPath SPIDER.zip
```

In Colab, upload `SPIDER.zip` and extract:

```python
from google.colab import files
files.upload()          # select SPIDER.zip

!unzip -q SPIDER.zip
```

## Step 2 — Set up runtime

Menu: **Runtime → Change runtime type → T4 GPU** (free). Then:

```python
%cd /content/SPIDER
!pip install -q timm albumentations
```

## Step 3 — Run experiments

Run everything (ordinal + CV + ablation):

```bash
!python improved_experiments.py --epochs 40
```

Or pick specific ones (e.g. only ordinal + ablation, which are fastest):

```bash
!python improved_experiments.py --experiments ordinal ablation --epochs 40
```

Estimated times on a free T4 GPU (40 epochs):
- **Ordinal**: ~40–60 min
- **Ablation** (2 runs): ~80–120 min
- **CV** (5 runs): ~3–5 h  ← run this overnight / in the background

## Step 4 — Download results

Everything lands in `results_improved/` and `figures_improved/`:

```python
!zip -r results_improved.zip results_improved figures_improved
from google.colab import files
files.download('results_improved.zip')
```

Copy `results_improved.zip` back into `C:\Users\srish\SPIDER\` and unzip it. Tell Claude the new numbers and it will wire them into the paper (new tables + figures).

## Local alternative

The same script runs on your own GPU (RTX 4050):

```bash
cd C:\Users\srish\SPIDER
.venv_cuda\Scripts\python.exe improved_experiments.py --epochs 40
```

---
**Note:** the ensemble experiment (EViT + MobileViT late fusion) already ran locally from your existing checkpoints — see `results/ensemble_evit_mvit.json`. It improves most per-pathology AUCs (Spondylolisthesis 0.847→0.857, LOW_endplate 0.722→0.737) but not pooled binary / Pfirrmann, so it's reported as an honest complementary analysis, not an inflated claim.
