"""
collect_features.py
===================
Gather feature rows for all patients in a retrain JSON (buffer + new_ood)
from several feature CSVs into one combined CSV.

After a CL cycle, the patients in cl_retrain.json have their features
spread across multiple CSVs (training CSV, step-2 CSV, step-3 CSV, ...).
This script finds each patient's row across those CSVs and writes them
all to a single CSV you can use for the next cycle's OOD detection.

Patient matching:
  - filename is derived from the JSON's 'image' field:
        .../NACT_12/nact_12_0001.nii.gz  ->  nact_12_0001
  - that filename is looked up in the 'filename' column of each CSV

If a patient is not found in ANY of the given CSVs, the script STOPS
and reports it (this means something is wrong, since the JSON was
built from these features).

Usage
-----
    python collect_features.py \
        --json cl_retrain.json \
        --csv  train_features.csv \
               step_two_features.csv \
               step_three_features.csv \
        --output combined_features.csv
"""

import argparse
import json
import os
import sys
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Collect feature rows for JSON patients from multiple CSVs."
    )
    p.add_argument("--json",   required=True,
                   help="Retrain JSON (cl_retrain.json) with 'image' paths.")
    p.add_argument("--csv",    required=True, nargs="+",
                   help="One or more feature CSV paths to search.")
    p.add_argument("--output", required=True,
                   help="Output combined CSV path.")
    p.add_argument("--key",    default="training",
                   help="Top-level JSON key holding the patient list "
                        "(default 'training').")
    return p.parse_args()


def filename_from_image(image_path: str) -> str:
    """.../NACT_12/nact_12_0001.nii.gz  ->  nact_12_0001"""
    return os.path.basename(image_path).replace(".nii.gz", "")


def main():
    args = parse_args()

    print("=" * 60)
    print("COLLECT FEATURES")
    print("=" * 60)
    print(f"  JSON   : {args.json}")
    print(f"  CSVs   : {len(args.csv)} files")
    for c in args.csv:
        print(f"           {c}")
    print(f"  Output : {args.output}")

    # ---- 1. Read patient filenames from JSON ----
    with open(args.json) as f:
        data = json.load(f)

    entries = data.get(args.key, [])
    if not entries:
        sys.exit(f"ERROR: no entries under key '{args.key}' in {args.json}")

    wanted = [filename_from_image(e["image"]) for e in entries]
    print(f"\n  Patients in JSON : {len(wanted)}")

    # ---- 2. Load and concatenate all CSVs into one lookup table ----
    frames = []
    for c in args.csv:
        if not os.path.exists(c):
            sys.exit(f"ERROR: CSV not found: {c}")
        df = pd.read_csv(c)
        if "filename" not in df.columns:
            sys.exit(f"ERROR: no 'filename' column in {c}")
        frames.append(df)
        print(f"  Loaded {len(df):>5} rows from {os.path.basename(c)}")

    all_feats = pd.concat(frames, ignore_index=True)
    # Drop duplicate filenames (keep first) just in case
    all_feats = all_feats.drop_duplicates(subset="filename", keep="first")
    lookup = all_feats.set_index("filename")
    print(f"  Total unique rows available : {len(lookup)}")

    # ---- 3. Find each wanted patient ----
    found_rows = []
    missing    = []
    for fn in wanted:
        if fn in lookup.index:
            row = lookup.loc[fn].copy()
            row["filename"] = fn        # restore the index column
            found_rows.append(row)
        else:
            missing.append(fn)

    # ---- 4. Stop if any patient is missing ----
    if missing:
        print(f"\n  ERROR: {len(missing)} patients not found in any CSV:")
        for m in missing[:20]:
            print(f"    {m}")
        if len(missing) > 20:
            print(f"    ... and {len(missing) - 20} more")
        sys.exit(1)

    # ---- 5. Build output CSV with same columns as input ----
    out_df = pd.DataFrame(found_rows)
    # Put 'filename' first, then the rest in original CSV order
    cols = ["filename"] + [c for c in all_feats.columns if c != "filename"]
    out_df = out_df[cols]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df.to_csv(args.output, index=False)

    print(f"\n{'='*60}")
    print(f"  SAVED -> {args.output}")
    print(f"  Rows  : {len(out_df)}  (all {len(wanted)} patients found)")
    print(f"  Cols  : {len(out_df.columns)}  (filename + "
          f"{len(out_df.columns)-1} features)")


if __name__ == "__main__":
    main()