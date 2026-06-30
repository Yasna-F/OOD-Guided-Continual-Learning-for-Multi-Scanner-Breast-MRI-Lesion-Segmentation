"""
run_update.py
=============
Day-to-day entry point for the incremental OOD pipeline.

Inputs
------
    --csv       CSV of extracted features for the new batch
    --json      JSON with filename->domain_id mapping for the new batch
    --registry  Path to the existing registry .pkl  (NEVER modified)

Outputs
-------
    ood_registry_<suffix>.pkl            new registry (only if new subgroups found)
    <output_dir>/assignments_<suffix>.json   per-patient subgroup assignments
    <output_dir>/results_<suffix>.csv        full results table
    <output_dir>/plots/                      summary figure

The input registry is NEVER touched. If new subgroups are found, the updated
registry is written to ood_registry_<suffix>.pkl alongside the original.
Pass that file as --registry on the next run.

Usage (minimal)
---------------
    python run_update.py \
        --csv  path/to/new_features.csv \
        --json path/to/new_images.json  \
        --registry path/to/ood_registry.pkl

Usage (full)
------------
    python run_update.py \
        --csv        path/to/new_features.csv  \
        --json       path/to/new_images.json   \
        --registry   path/to/ood_registry.pkl  \
        --output_dir path/to/outputs/          \
        --suffix     batch01                   \
        --min_size   5                         \
        --threshold  thr_mean3std              \
        --sil        0.25                      \
        --no_plot
"""

import argparse
import os
import pandas as pd

from utilities import (
    DOMAIN_INFO,
    load_features_with_domain,
    load_registry,
    save_registry,
    save_patient_assignments,
    get_early_feature_cols,
    run_ood_update,
    plot_results,
    print_subgroup_summary,
    print_ood_rate_per_scanner,
)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Incremental OOD detection -- update registry with new images."
    )
    p.add_argument("--csv",        required=True,
                   help="CSV of extracted features for the new batch.")
    p.add_argument("--json",       required=True,
                   help="JSON with filename->domain_id mapping for the new batch.")
    p.add_argument("--registry",   required=True,
                   help="Existing ood_registry.pkl. This file is NEVER modified.")
    p.add_argument("--output_dir", default=None,
                   help="Directory for assignments JSON, results CSV, and plots. "
                        "Defaults to the same folder as --registry.")
    p.add_argument("--suffix",     default="update",
                   help="Suffix appended to all output filenames (default: 'update').")
    p.add_argument("--min_size",   type=int,   default=5,
                   help="Min patients to form a new subgroup (default: 5).")
    p.add_argument("--threshold",  default="thr_mean3std",
                   choices=["thr_mean3std", "threshold"],
                   help="Which threshold to use (default: thr_mean3std).")
    p.add_argument("--sil",        type=float, default=0.25,
                   help="Silhouette score gate for new clusters (default: 0.25).")
    p.add_argument("--no_plot",    action="store_true",
                   help="Skip generating the summary plot.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #

def main():
    args = parse_args()

    registry_abs  = os.path.abspath(args.registry)
    registry_dir  = os.path.dirname(registry_abs)
    registry_stem = os.path.splitext(registry_abs)[0]   # strip .pkl

    output_dir = args.output_dir or registry_dir
    os.makedirs(output_dir, exist_ok=True)
    plot_dir = os.path.join(output_dir, "plots")

    # Output file paths
    assignments_path  = os.path.join(output_dir, f"assignments_{args.suffix}.json")
    results_csv_path  = os.path.join(output_dir, f"results_{args.suffix}.csv")
    # New registry sits next to the original: ood_registry_batch01.pkl
    new_registry_path = f"{registry_stem}_{args.suffix}.pkl"

    print("=" * 60)
    print("OOD PIPELINE - run_update.py")
    print("=" * 60)
    print(f"  CSV                : {args.csv}")
    print(f"  JSON               : {args.json}")
    print(f"  Registry (input)   : {args.registry}  <- never modified")
    print(f"  Registry (new)     : {new_registry_path}")
    print(f"  Output dir         : {output_dir}")
    print(f"  Suffix             : {args.suffix}")

    # ------------------------------------------------------------------ #
    # 1. Load new data
    # ------------------------------------------------------------------ #
    print("\n[1/4] Loading new data...")
    test_df = load_features_with_domain(args.csv, args.json)
    test_df = test_df.sort_values("domain_id").reset_index(drop=True)

    test_df["scanner_model"] = test_df["domain_id"].map(
        lambda d: DOMAIN_INFO.get(d, {}).get("scanner_model", "Unknown")
    )
    test_df["manufacturer"] = test_df["domain_id"].map(
        lambda d: DOMAIN_INFO.get(d, {}).get("manufacturer", "Unknown")
    )

    feat_cols = get_early_feature_cols(test_df)
    print(f"  Patients        : {len(test_df)}")
    print(f"  Feature columns : {len(feat_cols)}")

    # ------------------------------------------------------------------ #
    # 2. Load existing registry  (loaded once, never written back)
    # ------------------------------------------------------------------ #
    print("\n[2/4] Loading registry...")
    registry, registry_raw = load_registry(args.registry)
    subgroups_before = set(registry.keys())

    # ------------------------------------------------------------------ #
    # 3. Run OOD detection + optional registry update
    # ------------------------------------------------------------------ #
    print("\n[3/4] Running OOD detection...")
    results_df, registry_log, registry = run_ood_update(
        test_df,
        feat_cols,
        registry,
        registry_raw,
        min_subgroup_size=args.min_size,
        use_thr=args.threshold,
        sil_threshold=args.sil,
    )

    registry_changed = set(registry.keys()) != subgroups_before

    # ------------------------------------------------------------------ #
    # 4. Save outputs
    # ------------------------------------------------------------------ #
    print("\n[4/4] Saving outputs...")

    # Per-patient assignments JSON
    save_patient_assignments(results_df, assignments_path)

    # Full results CSV
    results_df.drop(columns=["features"], errors="ignore").to_csv(
        results_csv_path, index=False
    )
    print(f"  Results CSV        -> {results_csv_path}")

    # New registry: only written if new subgroups were discovered
    if registry_changed:
        new_subgroups = sorted(set(registry.keys()) - subgroups_before)
        save_registry(registry, registry_raw, new_registry_path)
        print(f"  Registry (new)     -> {new_registry_path}")
        print(f"  New subgroups      : {new_subgroups}")
        print(f"  Pass '--registry {new_registry_path}' on the next run.")
    else:
        print(f"  Registry           : no new subgroups found, nothing written.")
        print(f"  Pass '--registry {args.registry}' on the next run (unchanged).")

    # Console summaries
    print_subgroup_summary(registry)
    print_ood_rate_per_scanner(results_df)

    # Optional plot
    if not args.no_plot:
        plot_results(results_df, registry, registry_log,
                     plot_dir=plot_dir, use_thr=args.threshold)

    print("\nDone.")
    print(f"  Assignments        : {assignments_path}")
    print(f"  Results CSV        : {results_csv_path}")
    if registry_changed:
        print(f"  Registry (new)     : {new_registry_path}")


if __name__ == "__main__":
    main()