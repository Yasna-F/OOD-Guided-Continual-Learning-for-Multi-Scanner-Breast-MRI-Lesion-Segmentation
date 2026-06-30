# OOD-Guided Continual Learning for Multi-Scanner Breast MRI Lesion Segmentation

Code for the five-stage continual learning (CL) pipeline described in the paper.

## Structure

```
data/
  splits/             input MONAI-style data lists (Training.json, BatchN.json, Held_out_test.json)
  cycles/              generated CycleN_training.json files (buffer + new OOD patients per cycle)
checkpoints/
  baseline.ckpt       step-0 model, trained on data/splits/Training.json
  cycle1.ckpt         step-1 model, fine-tuned on data/cycles/Cycle1_training.json (batch1)
  cycle2.ckpt         step-2 model, fine-tuned on data/cycles/Cycle2_training.json (batch2)
01_baseline_training/
  training.py           trains the initial 3D U-Net on Training.json -> baseline.ckpt
02_feature_extraction/
  model_features.py     extracts multi-level encoder features for a given data list + checkpoint
03_ood_buffer/
  build_initial_registry.py   builds the initial per-subgroup Gaussian registry (run once)
  run_update.py               [to be added] OOD detection + clustering for each new batch
  compute_criteria_score.py   scores old patients for buffer selection (representativeness/diversity/discriminability)
  build_json.py                merges scored buffer + new OOD patients into a CycleN_training.json
  collect_features.py          merges feature CSVs across cycles
  collect_assignments.py       merges assignment JSONs across cycles
  utilities.py                  shared helper functions
  README.md                     stage-specific usage
04_retrain/
  retrain.py             fine-tunes on buffer + new pseudo-domain for one CL cycle
```

## Pipeline order

```
01_baseline_training  -->  02_feature_extraction  -->  03_ood_buffer  -->  04_retrain
     (once)                  (each batch)            (each batch)        (each batch)
```

1. **01_baseline_training** — train the initial model on `data/splits/Training.json`.
2. **02_feature_extraction** — extract encoder features for the next batch (e.g. `data/splits/Batch1.json`) using `checkpoints/baseline.ckpt`.
3. **03_ood_buffer** — run the OOD detection, clustering, and buffer-selection scripts in order (see `03_ood_buffer/README.md`) to produce a new `data/cycles/CycleN_training.json`.
4. **04_retrain** — fine-tune on `CycleN_training.json` to produce the next cycle's checkpoint, then repeat from step 2 for the next batch.

## Environment

See `requirements.txt`. Paths inside scripts marked `# EDIT THESE PATHS` need to be set to your local data locations before running.
