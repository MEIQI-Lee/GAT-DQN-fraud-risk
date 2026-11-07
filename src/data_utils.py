"""Elliptic dataset loading, preprocessing and temporal splitting.

Dataset: Elliptic Bitcoin Transaction Graph (Weber et al., 2019)
  - 203,769 transaction nodes, 234,355 payment-flow edges (directed)
  - 165 anonymized features per node (93 local + 72 aggregated)
  - 49 timesteps (~2 weeks each); labels: 4,545 illicit / 42,019 licit /
    157,205 unknown (unknown nodes are used for message passing only)

Split protocol (out-of-time, mimics production deployment):
  timesteps  1..30 -> train | 31..34 -> validation | 35..49 -> test
Only *labeled* nodes contribute to losses and metrics.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from src import config


def _read_raw(raw_dir: Path):
    required = [
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
        "elliptic_txs_features.csv",
    ]
    for name in required:
        if not (raw_dir / name).exists():
            raise FileNotFoundError(
                f"{raw_dir / name} not found. Download the 'Elliptic Data Set' "
                f"from Kaggle (ellipticco/elliptic-data-set) and place the three "
                f"CSV files under data/raw/."
            )
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    return classes, edges, features


def build_dataset(force_rebuild: bool = False, protocol: str = "temporal") -> Data:
    """Build (or load cached) PyG Data object.

    protocol="temporal":  train/val from timesteps 1..34 (stratified 85/15),
                          test on 35..49 — the deployment-faithful protocol.
    protocol="random":    stratified random 70/15/15 over ALL labeled nodes
                          (protocol used by some public notebooks; reported
                          only as a reference number, see README).
    """
    assert protocol in ("temporal", "random")
    cache = config.DATA_PROCESSED / f"elliptic_graph_{protocol}.pt"
    if cache.exists() and not force_rebuild:
        payload = torch.load(cache, weights_only=True)
        return _to_data(payload)

    classes, edges, features = _read_raw(config.DATA_RAW)

    # ---- node indexing ----
    tx_ids = features[0].to_numpy()
    tx2idx = {tx: i for i, tx in enumerate(tx_ids)}
    n_nodes = len(tx_ids)

    # ---- labels: '1' -> illicit(1), '2' -> licit(0), 'unknown' -> -1 ----
    y = np.full(n_nodes, -1, dtype=np.int64)
    cls_map = {"1": 1, "2": 0}
    for tx, c in zip(classes["txId"].to_numpy(), classes["class"].to_numpy()):
        if c in cls_map:
            y[tx2idx[tx]] = cls_map[c]

    # ---- features: col0 = txId, col1 = timestep, col2.. = 165 features ----
    timestep = features[1].to_numpy(dtype=np.int64)
    x = features.iloc[:, 2:].to_numpy(dtype=np.float32)

    # ---- edges (keep directed: money flow direction is informative) ----
    src = edges["txId1"].map(tx2idx).to_numpy()
    dst = edges["txId2"].map(tx2idx).to_numpy()
    edge_index = torch.from_numpy(np.stack([src, dst])).long()

    # ---- masks (labeled nodes only) ----
    from sklearn.model_selection import train_test_split
    labeled = y >= 0
    if protocol == "temporal":
        # train/val: stratified random split inside timesteps 1..TRAIN_END_TS
        # test:      strictly later timesteps (out-of-time)
        train_pool = (timestep <= config.TRAIN_END_TS) & labeled
        test_mask = (timestep > config.TRAIN_END_TS) & labeled
        pool_idx = np.nonzero(train_pool)[0]
        tr_idx, val_idx = train_test_split(
            pool_idx, test_size=config.VAL_FRACTION,
            random_state=config.SEED, stratify=y[pool_idx],
        )
    else:
        # random reference protocol: 70/15/15 stratified over ALL labeled nodes
        test_mask = np.zeros(n_nodes, dtype=bool)
        pool_idx = np.nonzero(labeled)[0]
        tr_idx, tmp_idx = train_test_split(
            pool_idx, test_size=0.30, random_state=config.SEED,
            stratify=y[pool_idx],
        )
        val_idx, te_idx = train_test_split(
            tmp_idx, test_size=0.50, random_state=config.SEED,
            stratify=y[tmp_idx],
        )
        test_mask[te_idx] = True

    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[tr_idx] = True
    val_mask[val_idx] = True

    # ---- standardize features, fit on *train* nodes only (no leakage) ----
    scaler = StandardScaler().fit(x[train_mask])
    x = scaler.transform(x).astype(np.float32)

    payload = {
        "x": torch.from_numpy(x),
        "edge_index": edge_index,
        "y": torch.from_numpy(y),
        "timestep": torch.from_numpy(timestep),
        "train_mask": torch.from_numpy(train_mask),
        "val_mask": torch.from_numpy(val_mask),
        "test_mask": torch.from_numpy(test_mask),
        "tx_ids": torch.from_numpy(tx_ids.astype(np.int64)),
    }
    torch.save(payload, cache)
    return _to_data(payload)


def _to_data(payload: dict) -> Data:
    data = Data(x=payload["x"], edge_index=payload["edge_index"], y=payload["y"])
    data.timestep = payload["timestep"]
    data.train_mask = payload["train_mask"]
    data.val_mask = payload["val_mask"]
    data.test_mask = payload["test_mask"]
    data.tx_ids = payload["tx_ids"]
    return data


def describe(data: Data) -> str:
    """Human-readable dataset summary used by scripts and reports."""
    lines = [
        f"nodes={data.num_nodes:,}, edges={data.num_edges:,}, features={data.num_features}",
    ]
    for name in ("train", "val", "test"):
        mask = getattr(data, f"{name}_mask")
        yy = data.y[mask]
        n_ill = int((yy == 1).sum())
        n_lic = int((yy == 0).sum())
        ts = data.timestep[mask]
        lines.append(
            f"{name:>5}: labeled={int(mask.sum()):>6,} | illicit={n_ill:>5,} "
            f"licit={n_lic:>6,} | illicit ratio={n_ill / max(n_ill + n_lic, 1):.4f} "
            f"| timesteps {int(ts.min())}-{int(ts.max())}"
        )
    return "\n".join(lines)
