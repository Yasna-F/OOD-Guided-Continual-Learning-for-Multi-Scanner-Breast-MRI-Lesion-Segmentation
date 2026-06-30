"""
collect_assignments.py
======================
Build one "old patients" assignments JSON for the next CL step, by pulling
each patient's assignment record from several per-step assignment JSONs.

Parallel to collect_features.py, but for assignment records instead of
feature rows.

The patient LIST comes from a retrain JSON (e.g. cl_step_two.json) — these
are exactly the patients the model learned in the previous step (buffer +
new_ood), which become the "old" patients for the next step's buffer.

The assignment RECORDS (assigned_group, min_dist, image_path, ...) are looked
up across the per-step assignment JSONs you pass in.

Matching key:
  filename, derived from the 'image' field:
      .../NACT_12/nact_12_0001.nii.gz  ->  nact_12_0001

If a patient from the retrain JSON is not found in ANY assignments JSON,
the script STOPS and reports it.

Usage
-----
    python collect_assignments.py \
        --retrain_json cl_step_two.json \
        --assignments  assignments_batch0.json assignments_batch1.json \
        --output       old_patients_assignments.json
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect assignment records for retrain-JSON patients "
                    "from multiple assignment JSONs."
    )
    p.add_argument("--retrain_json", required=True,
                   help="Retrain JSON (e.g. cl_step_two.json) listing the "
                        "patients that are 'old' for the next step.")
    p.add_argument("--assignments",  required=True, nargs="+",
                   help="One or more assignment JSONs to search for records.")
    p.add_argument("--output",       required=True,
                   help="Output combined assignments JSON path.")
    p.add_argument("--retrain_key",  default="training",
                   help="Top-level key in the retrain JSON holding the "
                        "patient list (default 'training').")
    return p.parse_args()


def filename_from_image(image_path: str) -> str:
    """.../NACT_12/nact_12_0001.nii.gz  ->  nact_12_0001"""
    return os.path.basename(image_path).replace(".nii.gz", "")


def load_records(path):
    """
    Load an assignments JSON. Handles two shapes:
      - a plain list of records
      - a dict with a 'training' (or similar) key holding the list
    Returns a list of record dicts.
    """
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # assignment JSONs from save_patient_assignments are plain lists,
        # but be forgiving if wrapped
        for key in ("training", "assignments", "data"):
            if key in data and isinstance(data[key], list):
                return data[key]
    sys.exit(f"ERROR: could not find a record list in {path}")


def main():
    args = parse_args()

    print("=" * 60)
    print("COLLECT ASSIGNMENTS")
    print("=" * 60)
    print(f"  Retrain JSON : {args.retrain_json}")
    print(f"  Assignments  : {len(args.assignments)} files")
    for a in args.assignments:
        print(f"                 {a}")
    print(f"  Output       : {args.output}")

    # ---- 1. Which patients do we need (from retrain JSON) ----
    with open(args.retrain_json) as f:
        retrain_data = json.load(f)
    entries = retrain_data.get(args.retrain_key, [])
    if not entries:
        sys.exit(f"ERROR: no entries under '{args.retrain_key}' "
                 f"in {args.retrain_json}")
    wanted = [filename_from_image(e["image"]) for e in entries]
    print(f"\n  Patients wanted : {len(wanted)}")

    # ---- 2. Build a lookup from all assignment JSONs ----
    lookup = {}
    for a in args.assignments:
        if not os.path.exists(a):
            sys.exit(f"ERROR: assignments JSON not found: {a}")
        recs = load_records(a)
        n_added = 0
        for r in recs:
            # derive filename: prefer existing 'filename', else from image_path
            fn = r.get("filename")
            if fn is None and "image_path" in r:
                fn = filename_from_image(r["image_path"])
            if fn is None and "image" in r:
                fn = filename_from_image(r["image"])
            if fn is None:
                continue
            # keep first occurrence (duplicates assumed not to happen)
            if fn not in lookup:
                lookup[fn] = r
                n_added += 1
        print(f"  Loaded {len(recs):>5} records from "
              f"{os.path.basename(a)}  ({n_added} new)")

    print(f"  Total unique records available : {len(lookup)}")

    # ---- 3. Pull each wanted patient's record ----
    collected = []
    missing   = []
    for fn in wanted:
        if fn in lookup:
            collected.append(lookup[fn])
        else:
            missing.append(fn)

    # ---- 4. Stop if any missing ----
    if missing:
        print(f"\n  ERROR: {len(missing)} patients not found in any "
              f"assignments JSON:")
        for m in missing[:20]:
            print(f"    {m}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
        sys.exit(1)

    # ---- 5. Save as a plain list (same shape as assignment JSONs) ----
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(collected, f, indent=2)

    # group breakdown
    from collections import Counter
    groups = Counter(r.get("assigned_group", "?") for r in collected)

    print(f"\n{'='*60}")
    print(f"  SAVED -> {args.output}")
    print(f"  Records : {len(collected)}  (all {len(wanted)} found)")
    print(f"  Groups  :")
    for g, n in sorted(groups.items()):
        print(f"    {g:<24} n={n}")


if __name__ == "__main__":
    main()