"""
build_retrain_json.py
=====================
Builds the retraining JSON from:
  1. OLD scored assignments JSON -- top-K per subgroup (buffer)
  2. NEW assignments JSON        -- ALL patients in newly formed subgroups

Output JSON
-----------
{
  "training": [
    {
      "image"         : "/path/to/image_0001.nii.gz",
      "label"         : "/path/to/label.nii.gz",
      "domain"        : 8,
      "manufacturer"  : "GE",
      "scanner_model" : "GENESIS_SIGNA",
      "assigned_group": "GENESIS_SIGNA",
      "source"        : "buffer" | "new_ood"
    },
    ...
  ]
}

Label path convention
---------------------
  image : .../images/DUKE_592/duke_592_0001.nii.gz
  label : .../segmentations/filledholes/duke_592.nii.gz

Usage
-----
    python build_retrain_json.py \
        --old_json   assignments_scored.json \
        --new_json   assignments_batch01.json \
        --labels_dir /container_workspace/raw_data/MAMA_MIA/segmentations/filledholes \
        --output     cl_retrain.json \
        --k          10
"""

import argparse
import json
import os
from collections import defaultdict


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Build retraining JSON: top-K buffer from old + all new OOD."
    )
    p.add_argument("--old_json",   required=True,
                   help="Scored assignments JSON (old patients).")
    p.add_argument("--new_json",   required=True,
                   help="Assignments JSON for the new batch (from run_update.py).")
    p.add_argument("--labels_dir", required=True,
                   help="Directory containing label .nii.gz files.")
    p.add_argument("--output",     required=True,
                   help="Output path for the retraining JSON.")
    p.add_argument("--k",          type=int, default=10,
                   help="Number of buffer exemplars per old subgroup (default 10).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# HELPERS
# --------------------------------------------------------------------------- #

def derive_label_path(image_path: str, labels_dir: str) -> str:
    filename = os.path.basename(image_path)
    case     = filename.replace("_0001.nii.gz", "")
    return os.path.join(labels_dir, case + ".nii.gz")


def make_entry(record: dict, source: str, labels_dir: str) -> dict:
    image_path = record.get("image_path", "")
    return {
        "image"         : image_path,
        "label"         : derive_label_path(image_path, labels_dir),
        "domain"        : record.get("domain_id",      None),
        "manufacturer"  : record.get("manufacturer",   None),
        "scanner_model" : record.get("scanner_model",  None),
        "assigned_group": record.get("assigned_group", None),
        "source"        : source,
    }


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    print("=" * 60)
    print("BUILD RETRAIN JSON")
    print("=" * 60)
    print(f"  Old JSON   : {args.old_json}")
    print(f"  New JSON   : {args.new_json}")
    print(f"  Labels dir : {args.labels_dir}")
    print(f"  Output     : {args.output}")
    print(f"  K          : {args.k} per old subgroup")

    # ------------------------------------------------------------------ #
    # 1. Load old assignments -- top-K per subgroup
    # ------------------------------------------------------------------ #
    with open(args.old_json) as f:
        old_records = json.load(f)

    has_scores = "score_combined" in old_records[0]
    if not has_scores:
        print("\n  Warning: 'score_combined' missing -- using original order.")
        print("  Run compute_criteria_scores.py first for score-based selection.")

    old_by_group = defaultdict(list)
    for r in old_records:
        old_by_group[r["assigned_group"]].append(r)

    print(f"\n  Old patients : {len(old_records)}")

    buffer_entries = []
    for grp_name, patients in sorted(old_by_group.items()):
        if has_scores:
            patients = sorted(
                patients,
                key=lambda r: r.get("score_combined") or 0.0,
                reverse=True,
            )
        selected = patients[: args.k]
        for r in selected:
            buffer_entries.append(make_entry(r, "buffer", args.labels_dir))
        n_avail = len(patients)
        print(f"    {grp_name:<24} selected={len(selected):>3} / {n_avail}")

    # ------------------------------------------------------------------ #
    # 2. Load new assignments -- ALL patients in NEW subgroups only
    # ------------------------------------------------------------------ #
    with open(args.new_json) as f:
        new_records = json.load(f)

    old_group_names    = set(old_by_group.keys())
    new_subgroup_names = sorted({
        r["assigned_group"] for r in new_records
        if r["assigned_group"] not in old_group_names
        and r["assigned_group"] != "OOD_POOL"
    })
    still_ood = sum(1 for r in new_records if r["assigned_group"] == "OOD_POOL")
    known_in_new = sorted({
        r["assigned_group"] for r in new_records
        if r["assigned_group"] in old_group_names
    })

    print(f"\n  New patients       : {len(new_records)}")
    print(f"  -> New subgroups   : {new_subgroup_names or 'none'}")
    print(f"  -> Known subgroups : {known_in_new}  (skipped)")
    print(f"  -> OOD_POOL        : {still_ood}  (skipped)")

    new_ood_entries = []
    for grp_name in new_subgroup_names:
        patients = [r for r in new_records if r["assigned_group"] == grp_name]
        for r in patients:
            new_ood_entries.append(make_entry(r, "new_ood", args.labels_dir))
        print(f"    {grp_name:<24} {len(patients)} patients -> training")

    # ------------------------------------------------------------------ #
    # 3. Save
    # ------------------------------------------------------------------ #
    training = buffer_entries + new_ood_entries

    output_data = {
        "training": training,
        "meta": {
            "k"                 : args.k,
            "old_subgroups"     : sorted(old_group_names),
            "new_subgroups"     : new_subgroup_names,
            "n_buffer"          : len(buffer_entries),
            "n_new_ood"         : len(new_ood_entries),
            "n_training"        : len(training),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    # ------------------------------------------------------------------ #
    # 4. Summary + label check
    # ------------------------------------------------------------------ #
    print(f"\n{'='*60}")
    print(f"  SAVED -> {args.output}")
    print(f"{'='*60}")
    print(f"  training   : {len(training)}")
    print(f"    buffer (old)  : {len(buffer_entries)}")
    print(f"    new OOD       : {len(new_ood_entries)}")

    missing = [e["label"] for e in training if not os.path.exists(e["label"])]
    if missing:
        print(f"\n  WARNING: {len(missing)} label files not found on disk.")
        for p in missing[:5]:
            print(f"    {p}")
        if len(missing) > 5:
            print(f"    ... and {len(missing)-5} more")
    else:
        print(f"\n  All {len(training)} label paths verified on disk.")


if __name__ == "__main__":
    main()