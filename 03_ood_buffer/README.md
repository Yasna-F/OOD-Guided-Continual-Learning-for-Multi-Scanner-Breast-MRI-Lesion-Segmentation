# OOD + Buffer Selection — Stage 3

```
build_initial_registry.py  ->  run_update.py  ->  compute_criteria_score.py  ->  build_json.py  ->  (04_retrain)
        (once)                  (each batch)          (each batch)              (each batch)
```

`collect_features.py` and `collect_assignments.py` are merge utilities used from cycle 2 onward,
when buffer features/assignments need to be pulled from more than one prior cycle's output files.

## 1. Initial registry (run once)

Edit paths at the top of the file, then:
```bash
python build_initial_registry.py
```
-> `ood_registry.pkl` + `assignments_initial.json`

## 2. Process new batch

> `run_update.py` is not yet in this repo — it wraps `run_ood_update()` from `utilities.py`.
> Expected interface once added:
```bash
python run_update.py \
    --csv      new_features.csv \
    --json     new_images.json \
    --registry ood_registry.pkl \
    --output_dir ./results/ \
    --suffix   batch01
```
-> updated registry + `assignments_batch01.json`

## 3. Score old patients for buffer

```bash
python compute_criteria_score.py \
    --json   assignments_initial.json \
    --csv    train_features.csv \
    --ckpt   ../checkpoints/baseline.ckpt \
    --output assignments_scored.json
```
Adds `score_combined` per patient. Run on **old** patients (not the new batch).

## 4. Build training JSON

```bash
python build_json.py \
    --old_json   assignments_scored.json \
    --new_json   ./results/assignments_batch01.json \
    --labels_dir /path/to/labels \
    --output     ../data/cycles/CycleN_training.json \
    --k          16
```
-> top-K buffer per old group + all new OOD patients. This is the file passed to `04_retrain/retrain.py`.

## Merge utilities (cycle 2+)

```bash
python collect_features.py --json <retrain_json> --csv <csv1> <csv2> ... --output merged_features.csv
python collect_assignments.py --retrain_json <retrain_json> --assignments <a1.json> <a2.json> ... --output merged_assignments.json
```

## Tips

- Features must use the **same preprocessing as training**.
- Evaluate **per scanner**, not aggregated.
- Next cycles: features from the step-1 model (fixed); discriminability from the latest checkpoint.
