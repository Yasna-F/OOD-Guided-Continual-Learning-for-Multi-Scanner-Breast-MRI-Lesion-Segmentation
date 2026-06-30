"""
utilities.py
============
All reusable functions for the incremental OOD detection pipeline
and multi-criteria buffer selection.

Contents
--------
  Constants        : DOMAIN_INFO, INITIAL_SUBGROUPS
  Feature helpers  : get_all_feature_cols, get_early_feature_cols
  Math             : mah_dist
  I/O              : load_features_with_domain, save_registry,
                     load_registry, save_patient_assignments
  Registry         : build_initial_registry, refit_subgroup
  Detection        : detect_ood
  Clustering       : check_pool_for_clusters
  Pool utils       : reeval_pool
  Main pipeline    : run_ood_update
  Criteria scoring : score_representativeness, score_diversity,
                     score_discriminability, compute_criteria_scores
  Plotting         : plot_results
  Console          : print_subgroup_summary, print_ood_rate_per_scanner
"""

import json
import pickle
import os
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.covariance import LedoitWolf
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# =========================================================================== #
# DOMAIN METADATA
# =========================================================================== #

DOMAIN_INFO = {
    8 : {"manufacturer": "GE",      "scanner_model": "GENESIS_SIGNA"},
    9 : {"manufacturer": "GE",      "scanner_model": "GENESIS_SIGNA"},
    13: {"manufacturer": "GE",      "scanner_model": "Optima MR450w"},
    14: {"manufacturer": "GE",      "scanner_model": "SIGNA HDx"},
    15: {"manufacturer": "GE",      "scanner_model": "Signa HDxt"},
    16: {"manufacturer": "GE",      "scanner_model": "Signa HDxt"},
    18: {"manufacturer": "GE",      "scanner_model": "Signa HDxt"},
    20: {"manufacturer": "GE",      "scanner_model": "Signa HDxt"},
    21: {"manufacturer": "GE",      "scanner_model": "Signa HDxt"},
    2 : {"manufacturer": "SIEMENS", "scanner_model": "Avanto"},
    3 : {"manufacturer": "SIEMENS", "scanner_model": "Avanto"},
    5 : {"manufacturer": "SIEMENS", "scanner_model": "Avanto"},
    22: {"manufacturer": "SIEMENS", "scanner_model": "Skyra"},
    24: {"manufacturer": "SIEMENS", "scanner_model": "Symphony"},
    26: {"manufacturer": "SIEMENS", "scanner_model": "TrioTim"},
    27: {"manufacturer": "SIEMENS", "scanner_model": "Verio"},
}

INITIAL_SUBGROUPS = {
    "GENESIS_SIGNA"    : [8, 9],
    "Signa_HDxt"       : [15, 16, 18, 20, 21],
    "SIGNA_HDx+Optima" : [13, 14],
}


# =========================================================================== #
# FEATURE COLUMN HELPERS
# =========================================================================== #

def get_all_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c.startswith("feature_")]


def get_early_feature_cols(df: pd.DataFrame, layer_limit: int = 96) -> list:
    """Feature columns for layers 1-2 (index < layer_limit)."""
    return [
        c for c in df.columns
        if c.startswith("feature_") and int(c.split("_")[1]) < layer_limit
    ]


# =========================================================================== #
# MATH
# =========================================================================== #

def mah_dist(features: np.ndarray, mu: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    """Vectorised Mahalanobis distance for a batch of row-vectors."""
    features = np.atleast_2d(features)
    diff = features - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv, diff))


# =========================================================================== #
# I/O
# =========================================================================== #

def load_features_with_domain(csv_path: str, json_path: str) -> pd.DataFrame:
    """
    Load a feature CSV and attach domain_id + image_path from the JSON.

    The JSON may have keys: 'training', 'validation', 'test', 'train'.
    Each entry must have:
        {'image': '/full/path/to/<filename>.nii.gz', 'domain': <int>}

    Adds two columns to the DataFrame:
        domain_id  : int domain identifier
        image_path : full path to the .nii.gz file
    """
    df = pd.read_csv(csv_path)
    with open(json_path) as f:
        data = json.load(f)

    fname_to_domain     = {}
    fname_to_image_path = {}

    for split in ("training", "validation", "test", "train"):
        for entry in data.get(split, []):
            basename = entry["image"].split("/")[-1].replace(".nii.gz", "")
            fname_to_domain[basename]     = entry["domain"]
            fname_to_image_path[basename] = entry["image"]

    df["domain_id"]  = df["filename"].map(fname_to_domain)
    df["image_path"] = df["filename"].map(fname_to_image_path)
    return df


def save_registry(registry: dict, registry_raw: dict, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump({"registry": registry, "registry_raw": registry_raw}, f)
    print(f"[save_registry] Saved -> {path}  ({len(registry)} subgroups)")


def load_registry(path: str) -> tuple:
    with open(path, "rb") as f:
        data = pickle.load(f)
    print(f"[load_registry] Loaded <- {path}  ({len(data['registry'])} subgroups)")
    return data["registry"], data["registry_raw"]


def save_patient_assignments(results_df: pd.DataFrame, output_path: str) -> None:
    """
    Save per-patient assignments to a JSON file.

    Fields saved (when present):
        filename, image_path, domain_id, scanner_model, manufacturer,
        is_ood, assigned_group, min_dist, nearest, threshold,
        score_representativeness, score_diversity,
        score_discriminability, score_combined
    """
    keep_cols = [
        "filename", "image_path",
        "domain_id", "scanner_model", "manufacturer",
        "is_ood", "assigned_group",
        "min_dist", "nearest", "threshold",
        # criteria scores (present only after compute_criteria_scores)
        "score_representativeness", "score_diversity",
        "score_discriminability", "score_combined",
    ]
    present = [c for c in keep_cols if c in results_df.columns]
    records = results_df[present].to_dict(orient="records")

    def _clean(v):
        if isinstance(v, (np.bool_,)):    return bool(v)
        if isinstance(v, (np.integer,)):  return int(v)
        if isinstance(v, (np.floating,)): return float(v)
        if v is None or (isinstance(v, float) and np.isnan(v)): return None
        return v

    records = [{k: _clean(v) for k, v in r.items()} for r in records]

    with open(output_path, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[save_patient_assignments] Saved -> {output_path}  ({len(records)} patients)")


# =========================================================================== #
# REGISTRY HELPERS
# =========================================================================== #

def _name_subgroup(meta_list: list, cycle_id: int) -> str:
    scanners = [m.get("scanner_model") for m in meta_list if m is not None]
    scanners = [s for s in scanners if s is not None]
    if scanners:
        top, count = Counter(scanners).most_common(1)[0]
        if count / len(scanners) >= 0.8:
            return f"{top}_C{cycle_id}"
    return f"NewDomain_C{cycle_id}"


def refit_subgroup(registry: dict, registry_raw: dict, name: str) -> None:
    """Re-fit LedoitWolf on the raw features of an existing subgroup in-place."""
    feats = np.array(registry_raw[name])
    lw = LedoitWolf(assume_centered=False)
    lw.fit(feats)
    dists = mah_dist(feats, lw.location_, lw.precision_)
    registry[name].update({
        "mu"          : lw.location_,
        "cov_inv"     : lw.precision_,
        "n"           : len(feats),
        "shrinkage"   : float(lw.shrinkage_),
        "threshold"   : float(np.percentile(dists, 95)),
        "thr_mean3std": float(dists.mean() + 3 * dists.std()),
    })


def build_initial_registry(
    train_df: pd.DataFrame,
    feat_cols: list,
    subgroup_def: dict,
) -> tuple:
    """
    Fit one LedoitWolf distribution per subgroup on training data.

    Parameters
    ----------
    train_df     : DataFrame with feature columns, domain_id, image_path
    feat_cols    : Feature column names to use
    subgroup_def : {subgroup_name: [domain_id, ...]}

    Returns
    -------
    registry, registry_raw
    """
    registry     = {}
    registry_raw = {}

    for grp, domains in subgroup_def.items():
        mask  = train_df["domain_id"].isin(domains)
        feats = train_df[mask][feat_cols].values
        lw    = LedoitWolf(assume_centered=False)
        lw.fit(feats)
        dists = mah_dist(feats, lw.location_, lw.precision_)

        registry[grp] = {
            "mu"                 : lw.location_,
            "cov_inv"            : lw.precision_,
            "threshold"          : float(np.percentile(dists, 95)),
            "thr_mean3std"       : float(dists.mean() + 3 * dists.std()),
            "n"                  : int(mask.sum()),
            "shrinkage"          : float(lw.shrinkage_),
            "cycle"              : 0,
            "domains"            : domains,
            "formed_from"        : {},
            "formed_from_domains": {},
        }
        registry_raw[grp] = feats.tolist()

        print(
            f"  {grp:<22}  n={mask.sum():>4}  "
            f"shrinkage={lw.shrinkage_:.4f}  "
            f"thr(95th)={registry[grp]['threshold']:.3f}  "
            f"thr(3std)={registry[grp]['thr_mean3std']:.3f}"
        )

    return registry, registry_raw


# =========================================================================== #
# OOD DETECTION
# =========================================================================== #

def detect_ood(z: np.ndarray, registry: dict, use_thr: str = "thr_mean3std") -> dict:
    """
    Assign a single feature vector to the nearest subgroup.

    Returns
    -------
    dict with: is_ood, min_dist, nearest, threshold, all_distances
    """
    z = np.atleast_2d(z)
    distances = {
        name: float(mah_dist(z, g["mu"], g["cov_inv"])[0])
        for name, g in registry.items()
    }
    nearest   = min(distances, key=distances.get)
    min_dist  = distances[nearest]
    threshold = registry[nearest][use_thr]

    return {
        "is_ood"       : min_dist > threshold,
        "min_dist"     : min_dist,
        "nearest"      : nearest,
        "threshold"    : threshold,
        "all_distances": distances,
    }


# =========================================================================== #
# CLUSTERING
# =========================================================================== #

def check_pool_for_clusters(
    pool_features: list,
    pool_meta: list,
    min_subgroup_size: int = 5,
    sil_threshold: float = 0.25,
    cycle: int = 0,
) -> tuple:
    """
    Run agglomerative clustering on the accumulated OOD pool.

    Returns
    -------
    new_subgroups, remaining_feats, remaining_meta
    """
    feats = np.array(pool_features)
    n     = len(feats)

    if n < 3:
        return [], pool_features, pool_meta

    print(f"\n  [check_pool_for_clusters] pool n={n}")

    scanners = [m.get("scanner_model", "Unknown") for m in pool_meta if m]
    domains  = [m.get("domain_id",     "Unknown") for m in pool_meta if m]
    print(f"  Scanner breakdown : {dict(Counter(scanners))}")
    print(f"  Domain  breakdown : {dict(Counter(domains))}")

    # Pool-internal Mahalanobis distance matrix
    lw_pool = LedoitWolf(assume_centered=False)
    lw_pool.fit(feats)
    cov_inv_pool = lw_pool.precision_

    D = np.zeros((n, n))
    for i in range(n):
        diff = feats - feats[i]
        D[i] = np.sqrt(np.einsum("ij,jk,ik->i", diff, cov_inv_pool, diff))
    np.fill_diagonal(D, 0)

    best_k, best_score, best_labels = 1, -1.0, None
    max_k = min(6, n - 1, max(2, n // 3))

    if max_k >= 2:
        for k in range(2, max_k + 1):
            model  = AgglomerativeClustering(
                n_clusters=k, metric="precomputed", linkage="average"
            )
            labels   = model.fit_predict(D)
            n_unique = len(set(labels))
            if n_unique < 2 or n_unique >= n:
                continue
            score = silhouette_score(D, labels, metric="precomputed")
            print(f"    k={k}  silhouette={score:.3f}")
            if score > best_score:
                best_k, best_score = k, score
                best_labels = labels.copy()

    print(f"  Best k={best_k}  sil={best_score:.3f}  gate={sil_threshold}")

    if best_score < sil_threshold or best_labels is None:
        print("  -> No cluster structure found; pool unchanged.")
        return [], pool_features, pool_meta

    print(f"  Cluster structure found (sil={best_score:.3f})")

    new_subgroups   = []
    remaining_feats = []
    remaining_meta  = []

    for k in range(best_k):
        mask    = best_labels == k
        g_feats = feats[mask]
        g_meta  = [pool_meta[i] for i, m in enumerate(mask) if m]

        g_scanners = Counter([m.get("scanner_model", "?") for m in g_meta if m])
        g_domains  = Counter([m.get("domain_id",     "?") for m in g_meta if m])
        print(f"\n    Cluster {k+1}: n={len(g_feats)}  scanners={dict(g_scanners)}")

        if len(g_feats) < min_subgroup_size:
            remaining_feats.extend(g_feats.tolist())
            remaining_meta.extend(g_meta)
            print(f"      -> too small ({len(g_feats)} < {min_subgroup_size}), back to pool")
            continue

        lw_new = LedoitWolf(assume_centered=False)
        lw_new.fit(g_feats)
        dists  = mah_dist(g_feats, lw_new.location_, lw_new.precision_)
        name   = _name_subgroup(g_meta, cycle + len(new_subgroups) + 1)

        new_subgroups.append({
            "mu"                 : lw_new.location_,
            "cov_inv"            : lw_new.precision_,
            "threshold"          : float(np.percentile(dists, 95)),
            "thr_mean3std"       : float(dists.mean() + 3 * dists.std()),
            "n"                  : len(g_feats),
            "shrinkage"          : float(lw_new.shrinkage_),
            "cycle"              : cycle + 1,
            "silhouette"         : float(best_score),
            "name"               : name,
            "formed_from"        : dict(g_scanners),
            "formed_from_domains": dict(g_domains),
            "raw_features"       : g_feats.tolist(),
        })
        print(f"      -> New subgroup '{name}'")

    return new_subgroups, remaining_feats, remaining_meta


# =========================================================================== #
# POOL RE-EVALUATION
# =========================================================================== #

def reeval_pool(
    pool_features: list,
    pool_meta: list,
    registry: dict,
    use_thr: str,
) -> tuple:
    """Remove from pool any patient now inside a known subgroup."""
    still_ood_feats, still_ood_meta = [], []
    n_now_id = 0
    for z, m in zip(pool_features, pool_meta):
        result = detect_ood(z, registry, use_thr)
        if result["is_ood"]:
            still_ood_feats.append(z)
            still_ood_meta.append(m)
        else:
            n_now_id += 1
    print(f"  [reeval_pool] {n_now_id} now ID | {len(still_ood_feats)} still OOD")
    return still_ood_feats, still_ood_meta


# =========================================================================== #
# MAIN PIPELINE
# =========================================================================== #

def run_ood_update(
    test_df: pd.DataFrame,
    feat_cols: list,
    registry: dict,
    registry_raw: dict,
    min_subgroup_size: int = 5,
    use_thr: str = "thr_mean3std",
    sil_threshold: float = 0.25,
) -> tuple:
    """
    Full OOD detection + optional registry update on new data.

    Steps
    -----
    1. Score every patient -> collect OOD pool
    2. Cluster OOD pool (silhouette-gated)
    3. Add new subgroups -> retroactively relabel
    4. Re-evaluate pool with updated registry

    Returns
    -------
    results_df, registry_log, registry
    """
    pool_features = []
    pool_meta     = []
    results       = []
    # Continue cycle numbering from the registry's current max, so a second
    # update produces _C2 not another _C1. (Was hardcoded to 0 before.)
    cycle         = max([g.get("cycle", 0) for g in registry.values()], default=0)

    registry_log = [{
        "cycle"         : 0,
        "patient_idx"   : 0,
        "n_subgroups"   : len(registry),
        "subgroup_names": list(registry.keys()),
        "event"         : "initial",
    }]

    print(f"\n{'='*60}")
    print(f"OOD UPDATE -- {len(test_df)} patients")
    print(f"{'='*60}")
    print(f"  Subgroups         : {list(registry.keys())}")
    print(f"  Min subgroup size : {min_subgroup_size}")
    print(f"  Silhouette gate   : {sil_threshold}")
    print(f"  Threshold type    : {use_thr}")

    # Pass 1: score every patient
    print(f"\n-- Pass 1: scoring {len(test_df)} patients --")
    for idx, row in test_df.iterrows():
        z    = row[feat_cols].values.astype(float)
        meta = {
            "scanner_model": row.get("scanner_model", None),
            "manufacturer" : row.get("manufacturer",  None),
            "domain_id"    : row.get("domain_id",     None),
            "filename"     : row.get("filename",      None),
            "image_path"   : row.get("image_path",    None),
        }
        result = detect_ood(z, registry, use_thr)
        result.update({
            "assigned_group": "OOD_POOL" if result["is_ood"] else result["nearest"],
            "patient_idx"   : idx,
            "filename"      : meta["filename"],
            "image_path"    : meta["image_path"],
            "scanner_model" : meta["scanner_model"],
            "manufacturer"  : meta["manufacturer"],
            "domain_id"     : meta["domain_id"],
            "n_subgroups"   : len(registry),
            "pool_size"     : len(pool_features),
            "cycle"         : 0,
            "features"      : z,     # internal only; not saved to JSON
        })
        results.append(result)
        if result["is_ood"]:
            pool_features.append(z)
            pool_meta.append(meta)

    n_id  = sum(1 for r in results if not r["is_ood"])
    n_ood = len(pool_features)
    print(f"  ID : {n_id}   OOD : {n_ood}")

    # Pass 2: cluster the OOD pool
    print(f"\n-- Pass 2: clustering OOD pool (n={n_ood}) --")
    new_subgroups, pool_features, pool_meta = check_pool_for_clusters(
        pool_features, pool_meta,
        min_subgroup_size=min_subgroup_size,
        sil_threshold=sil_threshold,
        cycle=cycle,
    )

    # Pass 3: update registry + retroactive relabelling
    if new_subgroups:
        for sg in new_subgroups:
            name      = sg.pop("name")
            raw_feats = sg.pop("raw_features")
            registry[name]     = sg
            registry_raw[name] = raw_feats
            cycle += 1
            print(
                f"\n  New subgroup '{name}' | "
                f"n={sg['n']}  shrinkage={sg['shrinkage']:.4f}  "
                f"thr={sg[use_thr]:.4f}  total={len(registry)}"
            )
            registry_log.append({
                "cycle"         : cycle,
                "patient_idx"   : len(test_df) - 1,
                "n_subgroups"   : len(registry),
                "subgroup_names": list(registry.keys()),
                "new_subgroup"  : name,
                "event"         : "new_subgroup",
            })

        print("\n-- Pass 3: retroactive relabelling --")
        n_relabelled = 0
        for prev in results:
            if prev["is_ood"]:
                new_eval = detect_ood(prev["features"], registry, use_thr)
                if not new_eval["is_ood"]:
                    prev.update({
                        "is_ood"        : False,
                        "min_dist"      : new_eval["min_dist"],
                        "nearest"       : new_eval["nearest"],
                        "threshold"     : new_eval["threshold"],
                        "assigned_group": new_eval["nearest"],
                        "all_distances" : new_eval["all_distances"],
                    })
                    n_relabelled += 1
        print(f"  Relabelled {n_relabelled} patients as ID")

        pool_features, pool_meta = reeval_pool(
            pool_features, pool_meta, registry, use_thr
        )
    else:
        print("\n  No new subgroups -- registry unchanged.")

    print(f"\n{'='*60}")
    print(f"  Patients    : {len(results)}")
    print(f"  Subgroups   : {len(registry)}")
    print(f"  Pool final  : {len(pool_features)}")

    return pd.DataFrame(results), registry_log, registry


# =========================================================================== #
# MULTI-CRITERIA BUFFER SCORING
# Based on: Zhuang et al. 2022 (Pattern Recognition) +
#           Bera et al. MICCAI 2023
# =========================================================================== #

def score_representativeness(dists: np.ndarray) -> np.ndarray:
    """
    Criterion 1: Representativeness
    --------------------------------
    Inverted Mahalanobis distance to the subgroup centroid.
    Accepts the min_dist values already stored in the assignments JSON.
    Samples closest to the centroid score highest.

    Returns scores in [0, 1] normalised within the subgroup.
    """
    scores = 1.0 / (1.0 + dists)          # invert: lower dist -> higher score
    dmin, dmax = scores.min(), scores.max()
    if dmax > dmin:
        scores = (scores - dmin) / (dmax - dmin)
    else:
        scores = np.ones_like(scores)
    return scores


def score_diversity(features: np.ndarray) -> np.ndarray:
    """
    Criterion 2: Diversity (greedy coreset)
    ----------------------------------------
    Iteratively selects the sample farthest from the already-selected
    set in feature space.  The selection ORDER becomes the diversity
    score: the first sample picked gets score 1.0 (most unique),
    the last gets score 0.0 (most redundant).

    This means every patient gets a score regardless of buffer size --
    you decide how many to keep later.

    Returns scores in [0, 1].
    """
    n = len(features)
    if n == 1:
        return np.array([1.0])

    # Euclidean pairwise distances (fast; features already in embedding space)
    from sklearn.metrics import pairwise_distances
    D = pairwise_distances(features, metric="euclidean")

    selected   = []
    remaining  = list(range(n))
    order      = []

    # Start with the sample closest to the global mean (most representative seed)
    mean_vec  = features.mean(axis=0)
    dists_to_mean = np.linalg.norm(features - mean_vec, axis=1)
    seed      = int(np.argmin(dists_to_mean))
    selected.append(seed)
    remaining.remove(seed)
    order.append(seed)

    while remaining:
        # For each remaining sample, find its min distance to the selected set
        min_dists = D[np.ix_(remaining, selected)].min(axis=1)
        best_idx  = int(np.argmax(min_dists))          # farthest from selected
        chosen    = remaining[best_idx]
        selected.append(chosen)
        remaining.remove(chosen)
        order.append(chosen)

    # Convert selection order to scores: first selected = 1.0, last = 0.0
    scores       = np.zeros(n)
    for rank, idx in enumerate(order):
        scores[idx] = 1.0 - rank / (n - 1) if n > 1 else 1.0
    return scores


def score_discriminability(
    image_paths: list,
    ckpt_path: str,
    roi_size: tuple = (192, 192, 64),
    sw_batch_size: int = 4,
    overlap: float = 0.25,
    device: str = "cuda",
) -> np.ndarray:
    """
    Criterion 3: Discriminability (model uncertainty)
    --------------------------------------------------
    Runs sliding window inference on each image using the trained UNet.
    Computes mean voxel entropy of the softmax output.
    High entropy = model is uncertain = high gradient influence during retraining.

    No ground truth needed (Bera et al. MICCAI 2023).

    Parameters
    ----------
    image_paths   : list of .nii.gz paths
    ckpt_path     : path to .ckpt checkpoint (PyTorch Lightning format)
    roi_size      : sliding window patch size
    sw_batch_size : patches per forward pass
    overlap       : sliding window overlap fraction
    device        : 'cuda' or 'cpu'

    Returns
    -------
    scores in [0, 1], one per image
    """
    import torch
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
    from monai.inferers import sliding_window_inference
    from monai.transforms import (
        Compose, LoadImaged, EnsureChannelFirstd,
        Spacingd, Orientationd, NormalizeIntensityd,
        SpatialPadd, EnsureTyped,
    )

    # ---- Load model ----
    model = UNet(
        spatial_dims  = 3,
        in_channels   = 1,
        out_channels  = 2,
        channels      = (32, 64, 128, 256, 320, 320),
        strides       = (2, 2, 2, 2, 2),
        num_res_units = 2,
        norm          = Norm.BATCH,
        dropout       = 0.15,
    ).to(device)

    ckpt       = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    # Strip Lightning prefix '_model.' if present
    state_dict = {
        k.replace("_model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("_model.") or "." in k
    }
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # ---- Preprocessing — must match training exactly ----
    preprocess = Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(
            keys=["image"],
            pixdim=(0.644, 0.644, 1.2),
            mode="bilinear"),
        NormalizeIntensityd(
            keys=["image"],
            nonzero=False, channel_wise=False),
        SpatialPadd(
            keys=["image"],
            spatial_size=[193, 192, 112],
            mode="constant"),
        EnsureTyped(keys=["image"]),
    ])

    entropy_scores = []
    with torch.no_grad():
        for img_path in image_paths:
            data = preprocess({"image": img_path})
            img  = data["image"].unsqueeze(0).to(device)    # (1,1,H,W,D)

            logits = sliding_window_inference(
                img, roi_size, sw_batch_size,
                model, overlap=overlap
            )
            probs   = torch.softmax(logits, dim=1)           # (1,2,H,W,D)
            eps     = 1e-8
            entropy = -(probs * (probs + eps).log()).sum(dim=1)  # (1,H,W,D)
            entropy_scores.append(float(entropy.mean().cpu()))

    scores = np.array(entropy_scores)
    dmin, dmax = scores.min(), scores.max()
    if dmax > dmin:
        scores = (scores - dmin) / (dmax - dmin)
    else:
        scores = np.ones_like(scores)
    return scores



# =========================================================================== #
# PLOTTING
# =========================================================================== #

GE_C  = "#fdae6b"
SIE_C = "#6baed6"


def plot_results(
    results_df: pd.DataFrame,
    registry: dict,
    registry_log: list,
    plot_dir: str,
    use_thr: str = "thr_mean3std",
) -> None:
    """Save a 2x2 summary figure to plot_dir/incremental_simulation_batch.png."""
    os.makedirs(plot_dir, exist_ok=True)

    scanner_order = (
        results_df.groupby("scanner_model")["min_dist"]
        .median().sort_values().index.tolist()
    )
    id_mask  = ~results_df["is_ood"]
    ood_mask =  results_df["is_ood"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel 1: distance stream
    ax = axes[0, 0]
    ax.plot(results_df.index, results_df["min_dist"],
            alpha=0.3, linewidth=0.6, color="steelblue")
    ax.scatter(results_df.index[id_mask],  results_df["min_dist"][id_mask],
               c="#2ca02c", s=8, alpha=0.5, label="ID",  zorder=3)
    ax.scatter(results_df.index[ood_mask], results_df["min_dist"][ood_mask],
               c="#d62728", s=8, alpha=0.5, label="OOD", zorder=3)
    for log in registry_log[1:]:
        ax.axvline(log["patient_idx"], color="black", linestyle="--",
                   linewidth=1.5, alpha=0.7)
        ax.text(log["patient_idx"] + 1,
                results_df["min_dist"].max() * 0.95,
                f"G{log['n_subgroups']}", fontsize=8, ha="left")
    ax.set_xlabel("Patient index")
    ax.set_ylabel("Min Mahalanobis Distance")
    ax.set_title("Distance Stream", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 2: OOD rate per scanner
    ax = axes[0, 1]
    colors = [
        GE_C if results_df[results_df["scanner_model"] == s]["manufacturer"].iloc[0] == "GE"
        else SIE_C for s in scanner_order
    ]
    x    = np.arange(len(scanner_order))
    bars = ax.bar(
        x,
        [results_df[results_df["scanner_model"] == s]["is_ood"].mean() * 100
         for s in scanner_order],
        0.5, color=colors, alpha=0.85, edgecolor="white",
    )
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(scanner_order, rotation=50, ha="right", fontsize=8)
    ax.set_ylabel("OOD Rate %")
    ax.set_title("OOD Rate per Scanner", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[
        mpatches.Patch(facecolor=GE_C,  alpha=0.85, label="GE"),
        mpatches.Patch(facecolor=SIE_C, alpha=0.85, label="SIEMENS"),
    ], fontsize=9)

    # Panel 3: pool size over time
    ax = axes[1, 0]
    ax.plot(results_df.index, results_df["pool_size"],
            color="steelblue", linewidth=1.5)
    ax.fill_between(results_df.index, results_df["pool_size"],
                    alpha=0.2, color="steelblue")
    for log in registry_log[1:]:
        ax.axvline(log["patient_idx"], color="red", linestyle="--", linewidth=1.5)
        ax.text(log["patient_idx"] + 1,
                results_df["pool_size"].max() * 0.95,
                f"New: {log['new_subgroup']}", fontsize=7, color="red", va="top")
    ax.set_xlabel("Patient index")
    ax.set_ylabel("OOD Pool Size")
    ax.set_title("OOD Pool Size Over Time", fontweight="bold")
    ax.grid(linestyle="--", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    # Panel 4: registry table
    ax = axes[1, 1]
    ax.axis("off")
    headers = ["Subgroup", "n", "Threshold", "Shrinkage", "Cycle"]
    rows = [
        [name, str(g["n"]),
         f"{g.get('thr_mean3std', g['threshold']):.3f}",
         f"{g['shrinkage']:.4f}", str(g["cycle"])]
        for name, g in registry.items()
    ]
    col_w        = [0.38, 0.10, 0.18, 0.18, 0.12]
    row_h        = 0.10
    hdr_y        = 0.92
    cycle_colors = ["#EEF4FF", "#FFF5E6", "#E8F5E9",
                    "#FFF3E0", "#F3E5F5", "#E0F2F1"]
    x_pos        = [sum(col_w[:i]) for i in range(len(col_w))]

    for j, (hdr, x) in enumerate(zip(headers, x_pos)):
        ax.add_patch(plt.Rectangle(
            (x, hdr_y), col_w[j], row_h,
            facecolor="#0F1F38", edgecolor="white", lw=0.5,
            transform=ax.transAxes))
        ax.text(x + col_w[j] / 2, hdr_y + row_h / 2, hdr,
                ha="center", va="center", fontsize=9,
                fontweight="bold", color="white", transform=ax.transAxes)

    for i, row in enumerate(rows):
        y  = hdr_y - (i + 1) * row_h
        bg = cycle_colors[min(i, len(cycle_colors) - 1)]
        for j, (val, x) in enumerate(zip(row, x_pos)):
            ax.add_patch(plt.Rectangle(
                (x, y), col_w[j], row_h,
                facecolor=bg, edgecolor="#D0D8E4", lw=0.5,
                transform=ax.transAxes))
            ax.text(x + col_w[j] / 2, y + row_h / 2, val,
                    ha="center", va="center", fontsize=8.5,
                    color="#1C2B3A", transform=ax.transAxes)

    ax.set_title("Final Registry State", fontweight="bold", fontsize=11, pad=10)

    plt.suptitle(
        "Incremental OOD Detection -- Batch Mode\nEarly Encoder Features (Layers 1-2)",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    out_path = os.path.join(plot_dir, "incremental_simulation_batch.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"[plot_results] Saved -> {out_path}")


# =========================================================================== #
# CONSOLE SUMMARIES
# =========================================================================== #

def print_subgroup_summary(registry: dict, domain_info: dict = DOMAIN_INFO) -> None:
    print("\n" + "=" * 60)
    print("SUBGROUP SUMMARY")
    print("=" * 60)
    for name, g in registry.items():
        if g["cycle"] == 0:
            print(f"\n  {name}  (initial, cycle 0)")
            print(f"    domains   : {g.get('domains', '?')}")
            print(f"    n         : {g['n']}")
            print(f"    threshold : {g['thr_mean3std']:.4f}")
        else:
            print(f"\n  {name}  (cycle {g['cycle']})")
            for sc, cnt in sorted(g.get("formed_from", {}).items(),
                                   key=lambda x: -x[1]):
                print(f"    {sc:<16} n={cnt} ({100*cnt/g['n']:.0f}%)")
            print(f"    n         : {g['n']}")
            print(f"    shrinkage : {g['shrinkage']:.4f}")
            print(f"    threshold : {g['thr_mean3std']:.4f}")


def print_ood_rate_per_scanner(results_df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("OOD RATE PER SCANNER")
    print("=" * 60)
    scanner_order = (
        results_df.groupby("scanner_model")["min_dist"]
        .median().sort_values().index.tolist()
    )
    print(f"\n  {'Scanner':<16} {'Manuf':>8} {'n':>4}  "
          f"{'ID':>5}  {'OOD':>5}  {'ID%':>7}  {'Avg dist':>9}")
    print("  " + "-" * 65)
    for s in scanner_order:
        sub   = results_df[results_df["scanner_model"] == s]
        n_id  = (~sub["is_ood"]).sum()
        n_ood = sub["is_ood"].sum()
        manuf = sub["manufacturer"].iloc[0]
        print(
            f"  {s:<16} {manuf:>8} {len(sub):>4}  "
            f"{n_id:>5}  {n_ood:>5}  "
            f"{100*n_id/len(sub):>6.1f}%  "
            f"{sub['min_dist'].mean():>9.3f}"
        )