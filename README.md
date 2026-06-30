# OOD-Guided Continual Learning for Multi-Scanner Breast MRI Lesion Segmentation

This repository contains the code accompanying the paper **"OOD-Guided Continual Learning for Multi-Scanner Breast MRI Lesion Segmentation"**, submitted to the MICCAI 2026 Deep-Brea3th Workshop.

We tackle a practical deployment problem: a breast MRI lesion segmentation model trained on one scanner manufacturer degrades when it meets data from a new, unseen scanner. Instead of assuming new domains are known in advance, our pipeline **detects them automatically** and adapts to them through a lightweight continual learning cycle — without forgetting what it already learned.

## 🧠 Key Features

* **Automatic OOD detection** — frozen-encoder Mahalanobis distance against per-subgroup Gaussians (Ledoit-Wolf shrinkage) flags incoming patients as in- or out-of-distribution, no manual domain labels required
* **Unsupervised pseudo-domain discovery** — agglomerative clustering (Ward linkage, silhouette-gated) groups OOD patients into coherent new subgroups
* **Multi-criteria buffer selection** — representativeness, diversity, and discriminability scoring selects a compact rehearsal buffer to prevent catastrophic forgetting
* **5-stage modular pipeline** — feature extraction → OOD detection → buffer scoring → cycle JSON construction → fine-tuning, each runnable independently
* **Built on MONAI + PyTorch Lightning** — 3D U-Net backbone, configurable patch size, AdamW + CosineAnnealingLR
* **Validated on MAMA-MIA** — 1,219 cases spanning GE, Siemens, and Philips scanners across four public cohorts (DUKE, ISPY1, ISPY2, NACT)

## 📊 Headline Result

When a new Philips scanner appears mid-deployment, one CL cycle lifts Philips Dice from **0.726 → 0.836** (Wilcoxon p = 0.0019), while GE and Siemens performance stays stable (BWT = −0.014) — adaptation to the new domain without forgetting the old ones.

## 📁 Repository Structure

```
data/
  splits/             input MONAI-style data lists (Training.json, BatchN.json, Held_out_test.json)
  cycles/             generated CycleN_training.json files (buffer + new OOD patients per cycle)
checkpoints/
  baseline.ckpt       step-0 model, trained on data/splits/Training.json
  cycle1.ckpt         step-1 model, fine-tuned on data/cycles/Cycle1_training.json
  cycle2.ckpt         step-2 model, fine-tuned on data/cycles/Cycle2_training.json
01_baseline_training/
  training.py         trains the initial 3D U-Net → baseline.ckpt
02_feature_extraction/
  model_features.py   extracts encoder features for a given batch + checkpoint
03_ood_buffer/
  build_initial_registry.py   builds the initial per-subgroup Gaussian registry (run once)
  run_update.py               OOD detection + clustering for each new batch
  compute_criteria_score.py   scores old patients for buffer selection
  build_json.py               merges scored buffer + new OOD patients into CycleN_training.json
  collect_features.py         merges feature CSVs across cycles
  collect_assignments.py      merges assignment JSONs across cycles
  utilities.py                shared helper functions
  README.md                   stage-specific usage
04_retrain/
  retrain.py           fine-tunes on buffer + new pseudo-domain for one CL cycle
```

## 🔁 Pipeline Order

```
01_baseline_training  →  02_feature_extraction  →  03_ood_buffer  →  04_retrain
      (once)                (each new batch)        (each new batch)    (each new batch)
```

1. Train the initial model on `data/splits/Training.json`.
2. Extract encoder features for the next batch (e.g. `Batch1.json`) using the latest checkpoint.
3. Run the OOD + buffer scripts in `03_ood_buffer/` (see its own README) to produce a new `CycleN_training.json`.
4. Fine-tune on that cycle JSON to get the next checkpoint, then repeat from step 2 for the next batch.

## ⚙️ Environment

```bash
pip install -r requirements.txt
```
Paths marked `# EDIT THESE PATHS` at the top of each script need to point to your local data before running.

## 💾 Checkpoints

`.ckpt` files are too large for git and are distributed via [GitHub Releases](../../releases) — see `checkpoints/README.md` for download links and what each checkpoint corresponds to.

## 📦 Dataset

Built on the public **MAMA-MIA** dataset. See the dataset's own release for access and terms.

