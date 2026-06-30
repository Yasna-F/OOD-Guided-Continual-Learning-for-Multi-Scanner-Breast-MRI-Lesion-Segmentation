"""
compute_criteria_scores.py
==========================
Add multi-criteria buffer selection scores to an assignments JSON.

Inputs
------
    assignments JSON  -- contains min_dist + image_path per patient
                         (from build_initial_registry or run_update)
    features CSV      -- needed only for diversity (pairwise distances)
    UNet checkpoint   -- needed only for discriminability (inference)

Output
------
    enriched JSON with four new fields per patient:
        score_representativeness  (from min_dist in JSON, no extra input)
        score_diversity           (from CSV features, pairwise distances)
        score_discriminability    (from checkpoint, sliding window entropy)
        score_combined

Criteria
--------
  1. Representativeness : inverted min_dist already in JSON
  2. Diversity          : greedy coreset on feature vectors from CSV
  3. Discriminability   : mean voxel entropy via UNet sliding window

All scores normalised to [0, 1] within each subgroup.

Usage
-----
    python compute_criteria_scores.py \
        --json    assignments_initial.json \
        --csv     all_levels_concatenated.csv \
        --ckpt    best_model.ckpt \
        --output  assignments_scored.json
"""

import argparse
import json
import numpy as np
import pandas as pd

from utilities import (
    get_early_feature_cols,
    save_patient_assignments,
    score_representativeness,
    score_diversity,
    score_discriminability,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute multi-criteria buffer scores for an assignments JSON."
    )
    p.add_argument("--json",     required=True,
                   help="Assignments JSON (from build_initial_registry or run_update). "
                        "Must contain 'min_dist' and 'image_path' per patient.")
    p.add_argument("--csv",      required=True,
                   help="Features CSV. Used for diversity (pairwise distances).")
    p.add_argument("--ckpt",     required=True,
                   help="UNet checkpoint .ckpt. Used for discriminability (inference).")
    p.add_argument("--output",   required=True,
                   help="Output path for the enriched JSON.")
    p.add_argument("--w_rep",    type=float, default=0.33,
                   help="Weight for representativeness (default 0.33).")
    p.add_argument("--w_div",    type=float, default=0.33,
                   help="Weight for diversity (default 0.33).")
    p.add_argument("--w_disc",   type=float, default=0.34,
                   help="Weight for discriminability (default 0.34).")
    p.add_argument("--roi_size", type=int, nargs=3, default=[192, 192, 64],
                   metavar=("H", "W", "D"),
                   help="Sliding window patch size (default 192 192 64).")
    p.add_argument("--sw_batch", type=int, default=4,
                   help="Patches per forward pass (default 4).")
    p.add_argument("--overlap",  type=float, default=0.25,
                   help="Sliding window overlap fraction (default 0.25).")
    p.add_argument("--device",   default="cuda",
                   help="'cuda' or 'cpu' (default cuda).")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    weights = (args.w_rep, args.w_div, args.w_disc)
    total   = sum(weights)
    if abs(total - 1.0) > 0.01:
        print(f"Warning: weights sum to {total:.3f}, normalising.")
        weights = tuple(w / total for w in weights)
    w1, w2, w3 = weights

    print("=" * 60)
    print("COMPUTE CRITERIA SCORES")
    print("=" * 60)
    print(f"  JSON    : {args.json}")
    print(f"  CSV     : {args.csv}  (diversity)")
    print(f"  Ckpt    : {args.ckpt}  (discriminability)")
    print(f"  Output  : {args.output}")
    print(f"  Weights : rep={w1:.2f}  div={w2:.2f}  disc={w3:.2f}")

    # ---- Load assignments JSON ----
    with open(args.json) as f:
        records = json.load(f)

    assign_df = pd.DataFrame(records)

    # Validate required fields
    missing = [c for c in ("min_dist", "image_path", "assigned_group", "filename")
               if c not in assign_df.columns]
    if missing:
        raise ValueError(
            f"Missing fields in JSON: {missing}. "
            "Re-run build_initial_registry.py or run_update.py to regenerate."
        )

    print(f"\n  Patients  : {len(assign_df)}")
    print(f"  Subgroups : {sorted(assign_df['assigned_group'].unique().tolist())}")

    # ---- Load CSV (diversity only) ----
    feat_df   = pd.read_csv(args.csv)
    feat_cols = get_early_feature_cols(feat_df)
    feat_df   = feat_df.set_index("filename")
    print(f"  CSV rows  : {len(feat_df)}  features: {len(feat_cols)}")

    # ---- Score per subgroup ----
    all_scored = []

    for grp_name, grp_df in assign_df.groupby("assigned_group"):
        grp_df = grp_df.copy()
        n      = len(grp_df)
        print(f"\n  [{grp_name}]  n={n}")

        # --- Criterion 1: Representativeness ---
        # min_dist is already in the JSON; just invert and normalise.
        dists = grp_df["min_dist"].values.astype(float)
        r1    = score_representativeness(dists)
        print(f"    Representativeness : mean={r1.mean():.3f}")

        # --- Criterion 2: Diversity (needs CSV) ---
        filenames  = grp_df["filename"].tolist()
        valid_mask = [fn in feat_df.index for fn in filenames]
        n_missing  = sum(not v for v in valid_mask)
        if n_missing:
            print(f"    Warning: {n_missing} patients missing from CSV, "
                  f"diversity score set to 0 for them")

        valid_fns = [fn for fn, ok in zip(filenames, valid_mask) if ok]
        feats     = feat_df.loc[valid_fns][feat_cols].values.astype(float)

        r2_full = np.zeros(n)
        if len(feats) >= 2:
            r2_valid = score_diversity(feats)
            valid_positions = [i for i, ok in enumerate(valid_mask) if ok]
            for pos, score in zip(valid_positions, r2_valid):
                r2_full[pos] = score
        elif len(feats) == 1:
            r2_full[valid_mask.index(True)] = 1.0
        print(f"    Diversity          : mean={r2_full.mean():.3f}")

        # --- Criterion 3: Discriminability (needs ckpt) ---
        image_paths = grp_df["image_path"].tolist()
        print(f"    Discriminability   : running inference ({n} images)...")
        r3 = score_discriminability(
            image_paths,
            args.ckpt,
            roi_size      = tuple(args.roi_size),
            sw_batch_size = args.sw_batch,
            overlap       = args.overlap,
            device        = args.device,
        )
        print(f"                         mean={r3.mean():.3f}")

        combined = w1 * r1 + w2 * r2_full + w3 * r3
        print(f"    Combined           : mean={combined.mean():.3f}")

        grp_df["score_representativeness"] = r1
        grp_df["score_diversity"]          = r2_full
        grp_df["score_discriminability"]   = r3
        grp_df["score_combined"]           = combined

        all_scored.append(grp_df)

    # ---- Save ----
    result_df = pd.concat(all_scored, ignore_index=True)
    save_patient_assignments(result_df, args.output)

    # Top-5 per subgroup
    print("\n" + "=" * 60)
    print("TOP-5 PER SUBGROUP (by combined score)")
    print("=" * 60)
    for grp_name, grp_df in result_df.groupby("assigned_group"):
        top5 = grp_df.nlargest(5, "score_combined")[
            ["filename",
             "score_representativeness",
             "score_diversity",
             "score_discriminability",
             "score_combined"]
        ]
        print(f"\n  {grp_name}")
        print(top5.to_string(index=False))

    print(f"\nDone -> {args.output}")


if __name__ == "__main__":
    main()