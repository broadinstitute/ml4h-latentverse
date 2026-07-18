"""Multimodal decoupling pipeline — a faithful DUPLICATE of the web app's
MultiLoReFT orchestration (app/multimodal_service.MultimodalService.
decouple_representations + app/comparison_service._process_multimodal), run in a
CANONICAL CPU REFERENCE CONFIG so the result is reproducible.

Parity note (verified): the web app runs MultiLoReFT on GPU-or-CPU ("auto") with
a content-addressed cache and *may* use best-of-N restarts. GPU floats diverge
from CPU beyond any tight tolerance, so parity is NOT defined against arbitrary
production runs. It is defined against this documented CPU reference config:

    device = cpu, TORCH_NUM_THREADS = 1 (env-overridable),
    torch.use_deterministic_algorithms(True), n_restarts = 1, pinned seed.

Under that config the decoupling is bit-exactly reproducible run-to-run (probed:
max abs run-to-run diff = 0.0 across all three subspaces on the test fixture).
"""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from latentverse import pipeline as _pl

logger = logging.getLogger(__name__)


# app/config.py mirror (same env names + defaults as Config.MULTILOREFT_*).
def _cfg_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _mm_max_epochs() -> int:
    return _cfg_int("MULTILOREFT_MAX_EPOCHS", 800)


def _mm_train_sample_cap() -> int:
    return _cfg_int("MULTILOREFT_TRAIN_SAMPLE_CAP", 20000)


MULTILOREFT_LR = 1e-3
MULTILOREFT_BATCH_SIZE = 256
MULTILOREFT_PRUNING_THRESHOLD = 0.1
MULTILOREFT_SHARED_RANK = 4
MULTILOREFT_SPECIFIC_RANK = 4
MULTILOREFT_STANDARDIZE = True

SUBSPACE_NAMES = ("modality1_specific", "modality2_specific", "shared")


def configure_deterministic_cpu() -> None:
    """Pin the CANONICAL CPU reference config. Idempotent; safe to call once per
    process before decoupling."""
    import torch

    threads = _cfg_int("TORCH_NUM_THREADS", 1)
    try:
        torch.set_num_threads(threads)
    except RuntimeError:
        pass
    try:
        torch.use_deterministic_algorithms(True)
    except Exception as exc:  # pragma: no cover - torch build dependent
        logger.warning("use_deterministic_algorithms unavailable: %s", exc)


def decouple_representations(
    rep1: np.ndarray,
    rep2: np.ndarray,
    shared_rank: int = MULTILOREFT_SHARED_RANK,
    specific_rank: int = MULTILOREFT_SPECIFIC_RANK,
    random_state: int = 42,
    standardize: bool = MULTILOREFT_STANDARDIZE,
) -> Dict[str, np.ndarray]:
    """Duplicate MultimodalService.decouple_representations for n_restarts=1 on
    CPU. Every op mirrors the web app site-for-site so the learned subspaces
    match under the canonical config."""
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from latentverse.multiloreft import MultiLoReFT

    configure_deterministic_cpu()
    device = torch.device("cpu")

    if rep1.shape[0] != rep2.shape[0]:
        raise ValueError(f"Sample count mismatch: rep1 has {rep1.shape[0]}, rep2 has {rep2.shape[0]}")

    n_samples = rep1.shape[0]
    dim1, dim2 = rep1.shape[1], rep2.shape[1]

    # app/multimodal_service.py:209-213 — per-feature standardize, then float32.
    if standardize:
        rep1 = (rep1 - rep1.mean(axis=0)) / (rep1.std(axis=0) + 1e-8)
        rep2 = (rep2 - rep2.mean(axis=0)) / (rep2.std(axis=0) + 1e-8)
        rep1 = np.ascontiguousarray(rep1, dtype=np.float32)
        rep2 = np.ascontiguousarray(rep2, dtype=np.float32)

    base_seed = int(random_state)
    # MultimodalService.__init__: self.batch_size = max(batch_size, 1024).
    batch_size = max(MULTILOREFT_BATCH_SIZE, 1024)
    max_epochs = _mm_max_epochs()
    train_cap = _mm_train_sample_cap()

    @contextlib.contextmanager
    def _seeded(seed: int):
        np_state = np.random.get_state()
        np.random.seed(seed)
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                yield
        finally:
            np.random.set_state(np_state)

    t1_full = torch.tensor(rep1, dtype=torch.float32)
    t2_full = torch.tensor(rep2, dtype=torch.float32)
    eval_loader = DataLoader(TensorDataset(t1_full, t2_full), batch_size=batch_size, shuffle=False)

    with _seeded(base_seed):
        t1 = torch.tensor(rep1, dtype=torch.float32)
        t2 = torch.tensor(rep2, dtype=torch.float32)

        if n_samples > train_cap:
            sample_idx = torch.randperm(n_samples)[:train_cap]
            t1_pool, t2_pool = t1[sample_idx], t2[sample_idx]
            pool_size = train_cap
        else:
            t1_pool, t2_pool = t1, t2
            pool_size = n_samples

        n_val = max(1, int(pool_size * 0.15))
        n_train = pool_size - n_val
        indices = torch.randperm(pool_size)
        train_idx, val_idx = indices[:n_train], indices[n_train:]

        train_loader = DataLoader(
            TensorDataset(t1_pool[train_idx], t2_pool[train_idx]),
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(t1_pool[val_idx], t2_pool[val_idx]),
            batch_size=batch_size,
            shuffle=False,
        )

        model = MultiLoReFT(
            input_dims=[dim1, dim2],
            shared_rank=shared_rank,
            specific_rank=specific_rank,
            staging=False,
            shared_R_mode="pad",
            pruning_threshold=MULTILOREFT_PRUNING_THRESHOLD,
            pruning=True,
            device=device,
        ).to(device)

        early_stopping_config = {
            "shared": {"patience": 15, "max_epochs": 150, "min_improvement_ratio": 5e-4},
            "private": {"patience": 15, "max_epochs": 150, "min_improvement_ratio": 5e-4},
            "joint": {"patience": 15, "max_epochs": max_epochs, "min_improvement_ratio": 5e-4},
        }

        model.train_projection(
            dataloader=train_loader,
            val_dataloader=val_loader,
            early_stopping_config=early_stopping_config,
            lr=MULTILOREFT_LR,
            epochs=max_epochs,
        )

        model.eval()
        with torch.no_grad():
            phis = model.forward([t1_full.to(device), t2_full.to(device)])
            components = model.decouple(phis)
            zm1 = components[0][0].cpu().numpy()
            zm2 = components[1][0].cpu().numpy()
            zs1 = components[0][1].cpu().numpy()
            zs2 = components[1][1].cpu().numpy()
            shared = (zs1 + zs2) / 2.0

    return {
        "modality1_specific": zm1,
        "modality2_specific": zm2,
        "shared": shared,
    }


def run_multimodal_pipeline(cfg: "_pl.PipelineConfig") -> Dict[str, Dict[str, Any]]:
    """Duplicate app/comparison_service._process_multimodal: align two rep files,
    decouple, and run ``cfg.test_type`` on each of the three subspaces.

    Returns ``{subspace_name: {label_or_mode: raw_metrics}}``."""
    _pl.pin_deterministic_defaults()
    if not cfg.representations2_path:
        raise ValueError("Multimodal mode requires --representations2.")
    if not cfg.rep_id_col or not cfg.rep2_id_col:
        raise ValueError(
            "Multimodal mode requires an id column for both representation files (--id-col and --rep2-id-col)."
        )

    # Build the rep1 side exactly like the unimodal path (merge labels + subsample).
    merged_df, selected_labels = _pl._load_and_merge(cfg)

    rep1_cols = merged_df.filter(regex="^latent_").columns.tolist()
    rep2_df = _pl.load_representation_file(cfg.representations2_path, cfg.rep2_id_col)
    if cfg.rep2_id_col not in rep2_df.columns:
        raise _pl._missing_column("Representation 2 id (--rep2-id-col)", cfg.rep2_id_col, rep2_df)
    rep2_cols = [
        c
        for c in rep2_df.columns
        if c not in {cfg.rep2_id_col, _pl._ROW_ID} and pd.api.types.is_numeric_dtype(rep2_df[c])
    ]
    if not rep1_cols or not rep2_cols:
        raise ValueError("Could not find numeric feature columns in one of the files.")

    if merged_df[cfg.rep_id_col].duplicated().any():
        raise ValueError(f"Representation 1 id column '{cfg.rep_id_col}' has duplicate values.")
    if rep2_df[cfg.rep2_id_col].duplicated().any():
        raise ValueError(f"Representation 2 id column '{cfg.rep2_id_col}' has duplicate values.")

    rep2_rename = {c: f"rep2_latent_{i}" for i, c in enumerate(rep2_cols)}
    rep2_aligned_cols = list(rep2_rename.values())
    rep2_features = rep2_df[[cfg.rep2_id_col, *rep2_cols]].rename(columns=rep2_rename)

    merged_df = merged_df.copy()
    merged_df[cfg.rep_id_col] = _pl._normalize_id_series(merged_df[cfg.rep_id_col])
    rep2_features[cfg.rep2_id_col] = _pl._normalize_id_series(rep2_features[cfg.rep2_id_col])
    aligned_df = pd.merge(
        merged_df,
        rep2_features,
        left_on=cfg.rep_id_col,
        right_on=cfg.rep2_id_col,
        how="inner",
        copy=False,
    )
    if len(aligned_df) == 0:
        raise ValueError("No rows remain after aligning the two representation files.")

    rep1_array = np.asarray(aligned_df[rep1_cols].values, dtype=np.float32)
    rep2_array = np.asarray(aligned_df[rep2_aligned_cols].values, dtype=np.float32)

    subspaces = decouple_representations(
        rep1_array,
        rep2_array,
        shared_rank=MULTILOREFT_SHARED_RANK,
        specific_rank=MULTILOREFT_SPECIFIC_RANK,
        random_state=cfg.random_seed,
        standardize=MULTILOREFT_STANDARDIZE,
    )

    labels_data = aligned_df[selected_labels] if selected_labels else None

    out: Dict[str, Dict[str, Any]] = {}
    for name in SUBSPACE_NAMES:
        arr = subspaces[name]
        subspace_df = pd.DataFrame({f"latent_{i}": arr[:, i] for i in range(arr.shape[1])})
        if labels_data is not None:
            for col in selected_labels:
                subspace_df[col] = labels_data[col].values

        if selected_labels:
            per_label: Dict[str, Any] = {}
            for label_col in selected_labels:
                per_label[label_col] = _pl._run_single_label(
                    subspace_df,
                    cfg.test_type,
                    label_col,
                    cfg.robustness_metric,
                    cfg.random_seed,
                    cfg.standardize,
                    cfg.num_clusters_override,
                    cfg.percent_removed,
                    cfg.noise_levels,
                )
            out[name] = per_label
        else:
            out[name] = {
                "Intrinsic (No Labels)": _pl._run_unlabeled(
                    subspace_df,
                    cfg.test_type,
                    cfg.robustness_metric,
                    cfg.random_seed,
                    cfg.standardize,
                    cfg.num_clusters_override,
                    cfg.noise_levels,
                )
            }
    return out
