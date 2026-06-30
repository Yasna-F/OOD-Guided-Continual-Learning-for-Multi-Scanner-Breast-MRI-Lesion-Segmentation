"""
build_initial_registry.py
=========================
Run ONCE to create the initial registry from training data.

Usage
-----
    python build_initial_registry.py

Edit the PATHS section below before running.
Produces:
    ood_registry.pkl          -- subgroup distributions
    assignments_initial.json  -- per-patient subgroup assignment + image_path
"""

import pandas as pd
from utilities import (
    INITIAL_SUBGROUPS,
    DOMAIN_INFO,
    load_features_with_domain,
    get_early_feature_cols,
    build_initial_registry,
    save_registry,
    save_patient_assignments,
    mah_dist,
)

# ---- EDIT THESE PATHS -------------------------------------------------------
TRAIN_CSV        = "/container_workspace/Code/CL/MAMA_MIA_Again/model_feature/extracted_features_training_avg/all_levels_concatenated.csv"
JSON_TRAIN       = "/container_workspace/Code/CL/MAMA_MIA_Again/step_one/step_one_p1.json"
REGISTRY_PATH    = "/container_workspace/Code/CL/MAMA_MIA_Again/model_feature/ood_registry.pkl"
ASSIGNMENTS_PATH = "./assignments_initial.json"
# -----------------------------------------------------------------------------

print("=" * 60)
print("BUILD INITIAL REGISTRY")
print("=" * 60)

# 1. Load training features + image paths
print("\nLoading training data...")
train_df  = load_features_with_domain(TRAIN_CSV, JSON_TRAIN)
feat_cols = get_early_feature_cols(train_df)    # layers 1-2 (index < 96)

train_df["scanner_model"] = train_df["domain_id"].map(
    lambda d: DOMAIN_INFO.get(d, {}).get("scanner_model", "Unknown")
)
train_df["manufacturer"] = train_df["domain_id"].map(
    lambda d: DOMAIN_INFO.get(d, {}).get("manufacturer", "Unknown")
)

print(f"  Samples         : {len(train_df)}")
print(f"  Feature columns : {len(feat_cols)}")
print(f"  Image paths OK  : {train_df['image_path'].notna().sum()}")

# 2. Fit LedoitWolf per initial subgroup
print("\nFitting subgroups...")
registry, registry_raw = build_initial_registry(train_df, feat_cols, INITIAL_SUBGROUPS)

# 3. Save registry
save_registry(registry, registry_raw, REGISTRY_PATH)

# 4. Assign every training patient directly to their known subgroup.
#    No OOD detection -- training patients are what the subgroups were built from.
#    Mahalanobis distance to own subgroup is recorded for reference only.
print("\nAssigning training patients to their known subgroups...")

# Reverse map: domain_id -> subgroup_name
domain_to_subgroup = {
    domain_id: grp
    for grp, domain_ids in INITIAL_SUBGROUPS.items()
    for domain_id in domain_ids
}

records = []
for _, row in train_df.iterrows():
    domain_id     = row.get("domain_id",     None)
    subgroup_name = domain_to_subgroup.get(domain_id, "UNKNOWN")
    z             = row[feat_cols].values.astype(float)

    if subgroup_name in registry:
        g    = registry[subgroup_name]
        dist = float(mah_dist(z, g["mu"], g["cov_inv"])[0])
        thr  = g["thr_mean3std"]
    else:
        dist = None
        thr  = None

    records.append({
        "filename"        : row.get("filename",      None),
        "image_path"      : row.get("image_path",    None),
        "domain_id"       : domain_id,
        "scanner_model"   : row.get("scanner_model", None),
        "manufacturer"    : row.get("manufacturer",  None),
        "assigned_group"  : subgroup_name,
        "min_dist"        : dist,
    })

results_df = pd.DataFrame(records)
save_patient_assignments(results_df, ASSIGNMENTS_PATH)
print(f"  {len(results_df)} training patients assigned.")

# Quick sanity check
unknown = (results_df["assigned_group"] == "UNKNOWN").sum()
if unknown:
    print(f"  Warning: {unknown} patients have domain_id not in INITIAL_SUBGROUPS")
else:
    print("  All patients mapped to a known subgroup.")

print("\nDone. Run run_update.py for new batches.")