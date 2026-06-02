from __future__ import annotations

from dataclasses import dataclass
import io
from pathlib import Path
import zipfile
from functools import lru_cache

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download

from .masks import random_missing_mask
from .normalization import StandardScaler


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    adjacency: torch.Tensor
    scaler: StandardScaler
    metadata: dict


class TrafficWindowDataset(Dataset):
    def __init__(self, full_x: np.ndarray, obs_x: np.ndarray, obs_mask: np.ndarray, extra_masks: dict[str, np.ndarray] | None = None):
        self.full_x = torch.tensor(full_x, dtype=torch.float32)
        self.obs_x = torch.tensor(obs_x, dtype=torch.float32)
        self.obs_mask = torch.tensor(obs_mask, dtype=torch.float32)
        self.extra_masks = {
            name: torch.tensor(mask, dtype=torch.float32)
            for name, mask in (extra_masks or {}).items()
        }

    def __len__(self) -> int:
        return self.full_x.shape[0]

    def __getitem__(self, idx: int) -> dict:
        full_x = self.full_x[idx]
        obs_x = self.obs_x[idx]
        obs_mask = self.obs_mask[idx]
        target_mask = 1.0 - obs_mask
        if target_mask.sum() <= 0:
            target_mask = torch.ones_like(target_mask)
        item = {
            "x_full": full_x,
            "x_obs": obs_x,
            "mask": obs_mask,
            "target_mask": target_mask,
        }
        for name, mask in self.extra_masks.items():
            item[name] = mask[idx]
        return item


def _build_ring_adjacency(num_nodes: int) -> np.ndarray:
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for i in range(num_nodes):
        adj[i, i] = 1.0
        adj[i, (i - 1) % num_nodes] = 1.0
        adj[i, (i + 1) % num_nodes] = 1.0
    degree = adj.sum(axis=1, keepdims=True)
    return adj / np.clip(degree, 1.0, None)


def _generate_toy_tensor(num_samples: int, seq_len: int, num_nodes: int, channels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    adj = _build_ring_adjacency(num_nodes)
    data = np.zeros((num_samples, seq_len, num_nodes, channels), dtype=np.float32)

    node_phase = np.linspace(0.0, 2.0 * np.pi, num_nodes, endpoint=False, dtype=np.float32)
    kernel = adj

    for sample_idx in range(num_samples):
        latent = rng.normal(0.0, 0.15, size=(seq_len, num_nodes)).astype(np.float32)
        latent[0] += np.sin(node_phase + rng.uniform(-0.5, 0.5)).astype(np.float32)
        for t in range(1, seq_len):
            smooth_prev = 0.72 * latent[t - 1]
            spatial_prev = 0.22 * (kernel @ latent[t - 1])
            seasonal = 0.12 * np.sin((2.0 * np.pi * t / seq_len) + node_phase)
            latent[t] += smooth_prev + spatial_prev + seasonal

        speed = 45.0 + 8.0 * latent
        occupancy = np.clip(0.25 + 0.08 * np.tanh(latent), 0.05, 0.95)
        flow = np.clip(speed * occupancy * 4.0 + rng.normal(0.0, 1.2, size=(seq_len, num_nodes)), 1.0, None)
        # Channel convention: 0=flow, 1=occupancy, 2=speed. This matches PEMS/ASTGNN.
        stacked = np.stack([flow, occupancy, speed], axis=-1)
        data[sample_idx] = stacked[:, :, :channels]

    return data, adj


def _window_time_series(series: np.ndarray, seq_len: int, max_samples: int, stride: int = 1) -> np.ndarray:
    if series.ndim != 3:
        raise ValueError("traffic series must have shape [time, nodes, channels].")
    if series.shape[0] < seq_len:
        raise ValueError("traffic series is shorter than seq_len.")
    windows = []
    for start in range(0, series.shape[0] - seq_len + 1, stride):
        windows.append(series[start : start + seq_len])
        if len(windows) >= max_samples:
            break
    if not windows:
        raise ValueError("no windows were created from traffic series.")
    return np.stack(windows, axis=0).astype(np.float32)


def _load_npz_array(path: Path) -> np.ndarray:
    payload = np.load(path, allow_pickle=True)
    for key in ("data", "x", "X", "arr_0"):
        if key in payload:
            data = payload[key]
            break
    else:
        data_keys = [key for key in payload.files if np.asarray(payload[key]).ndim in {2, 3}]
        if not data_keys:
            raise ValueError(f"no 2D/3D data array found in {path}")
        data = payload[data_keys[0]]
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f"expected [time,nodes,channels] data in {path}, got shape {data.shape}")
    return data


def _load_npz_array_from_bytes(payload_bytes: bytes, source_name: str) -> np.ndarray:
    payload = np.load(io.BytesIO(payload_bytes), allow_pickle=True)
    for key in ("data", "x", "X", "arr_0"):
        if key in payload:
            data = payload[key]
            break
    else:
        data_keys = [key for key in payload.files if np.asarray(payload[key]).ndim in {2, 3}]
        if not data_keys:
            raise ValueError(f"no 2D/3D data array found in {source_name}")
        data = payload[data_keys[0]]
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[..., None]
    if data.ndim != 3:
        raise ValueError(f"expected [time,nodes,channels] data in {source_name}, got shape {data.shape}")
    return data


def _edge_csv_to_adjacency(text: str, num_nodes: int) -> np.ndarray:
    adj = np.eye(num_nodes, dtype=np.float32)
    rows = np.genfromtxt(io.StringIO(text), delimiter=",", names=True, dtype=None, encoding="utf-8")
    if rows.shape == ():
        rows = np.asarray([rows])
    names = rows.dtype.names or ()
    if not {"from", "to"}.issubset(set(names)):
        return _build_ring_adjacency(num_nodes)
    for row in rows:
        source = int(row["from"])
        target = int(row["to"])
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            adj[source, target] = 1.0
            adj[target, source] = 1.0
    degree = adj.sum(axis=1, keepdims=True)
    return adj / np.clip(degree, 1.0, None)


def _load_adjacency(root: Path, num_nodes: int) -> tuple[np.ndarray, bool]:
    for name in ("adj.npy", "adjacency.npy", "adj_mx.npy", "graph.npy"):
        path = root / name
        if path.exists():
            adj = np.asarray(np.load(path, allow_pickle=True), dtype=np.float32)
            if adj.shape == (num_nodes, num_nodes):
                degree = adj.sum(axis=1, keepdims=True)
                return adj / np.clip(degree, 1.0, None), False
    for name in ("adj.csv", "adjacency.csv", "distance.csv"):
        path = root / name
        if path.exists():
            try:
                adj = np.loadtxt(path, delimiter=",", dtype=np.float32)
                if adj.shape == (num_nodes, num_nodes):
                    degree = adj.sum(axis=1, keepdims=True)
                    return adj / np.clip(degree, 1.0, None), False
            except ValueError:
                text = path.read_text(encoding="utf-8")
                return _edge_csv_to_adjacency(text, num_nodes), False
    return _build_ring_adjacency(num_nodes), True


def _find_zip_entry(zf: zipfile.ZipFile, suffix: str) -> zipfile.ZipInfo | None:
    suffix = suffix.replace("\\", "/").lower()
    for entry in zf.infolist():
        if entry.filename.replace("\\", "/").lower().endswith(suffix):
            return entry
    return None


def _load_pems08_from_zip(zip_path: Path, num_nodes: int) -> tuple[np.ndarray, np.ndarray, dict]:
    with zipfile.ZipFile(zip_path, "r") as zf:
        data_entry = (
            _find_zip_entry(zf, "data/PEMS08/PEMS08.npz")
            or _find_zip_entry(zf, "PEMS08.npz")
            or _find_zip_entry(zf, "pems08.npz")
        )
        if data_entry is None:
            raise ValueError(f"no PEMS08 npz entry found in {zip_path}")
        series = _load_npz_array_from_bytes(zf.read(data_entry), data_entry.filename)

        csv_entry = _find_zip_entry(zf, "data/PEMS08/PEMS08.csv") or _find_zip_entry(zf, "PEMS08.csv")
        if csv_entry is not None:
            adj = _edge_csv_to_adjacency(zf.read(csv_entry).decode("utf-8"), num_nodes)
            adjacency_fallback = False
        else:
            adj = _build_ring_adjacency(num_nodes)
            adjacency_fallback = True
    metadata = {
        "zip_path": str(zip_path),
        "zip_data_entry": data_entry.filename,
        "adjacency_fallback_ring": adjacency_fallback,
    }
    return series, adj, metadata


def _find_real_pems08_file(root: Path) -> Path | None:
    if not root.exists():
        return None
    candidates = [
        root / "PEMS08.npz",
        root / "pems08.npz",
        root / "PEMSD8.npz",
        root / "data.npz",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(root.glob("*.npz"))
    return matches[0] if matches else None


def _load_pems08_debug_splits(dataset_cfg: dict, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict] | None:
    root = Path(dataset_cfg.get("root", "data/raw/pems08"))
    if not root.is_absolute():
        root = Path(__file__).resolve().parents[1] / root
    zip_path_value = dataset_cfg.get("zip_path")
    zip_metadata = None
    data_path = None
    if zip_path_value:
        zip_path = Path(zip_path_value)
        if not zip_path.is_absolute():
            zip_path = Path(__file__).resolve().parents[1] / zip_path
        if zip_path.exists():
            series, adj, zip_metadata = _load_pems08_from_zip(zip_path, int(dataset_cfg["nodes"]))
        else:
            return None
    else:
        data_path = _find_real_pems08_file(root)
        if data_path is None:
            return None
        series = _load_npz_array(data_path)

    requested_nodes = int(dataset_cfg["nodes"])
    requested_channels = int(dataset_cfg["channels"])
    if series.shape[1] < requested_nodes:
        raise ValueError(f"PEMS08 file has only {series.shape[1]} nodes, requested {requested_nodes}.")
    if series.shape[2] < requested_channels:
        raise ValueError(f"PEMS08 file has only {series.shape[2]} channels, requested {requested_channels}.")
    series = series[:, :requested_nodes, :requested_channels]

    total_windows = int(dataset_cfg["train_samples"]) + int(dataset_cfg["val_samples"]) + int(dataset_cfg["test_samples"])
    windows = _window_time_series(series, int(dataset_cfg["seq_len"]), total_windows)
    train_end = int(dataset_cfg["train_samples"])
    val_end = train_end + int(dataset_cfg["val_samples"])
    train_x = windows[:train_end]
    val_x = windows[train_end:val_end]
    test_x = windows[val_end:val_end + int(dataset_cfg["test_samples"])]
    if len(test_x) == 0:
        raise ValueError("not enough PEMS08 windows for requested debug split.")

    if zip_metadata is None:
        adj, fallback_adj = _load_adjacency(root, requested_nodes)
    else:
        fallback_adj = bool(zip_metadata["adjacency_fallback_ring"])
    metadata = {
        "dataset_name": "pems08_debug",
        "fallback_used": False,
        "real_data_path": str(data_path) if data_path is not None else None,
        "zip_path": None if zip_metadata is None else zip_metadata["zip_path"],
        "zip_data_entry": None if zip_metadata is None else zip_metadata["zip_data_entry"],
        "adjacency_fallback_ring": fallback_adj,
        "source_shape": list(series.shape),
        "seed": seed,
    }
    return train_x, val_x, test_x, adj, metadata


def _normalize_splits(train_x: np.ndarray, val_x: np.ndarray, test_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler.fit(train_x)
    return scaler.transform(train_x), scaler.transform(val_x), scaler.transform(test_x), scaler


def _metrla_feature_columns() -> list[str]:
    cols = []
    for step in range(-11, 1):
        prefix = f"x_t{step:+d}"
        for channel in range(2):
            cols.append(f"{prefix}_d{channel}")
    for step in range(1, 13):
        prefix = f"y_t+{step}"
        for channel in range(2):
            cols.append(f"{prefix}_d{channel}")
    return cols


def _load_metrla_hf_split(split_name: str, max_windows: int) -> tuple[np.ndarray, dict]:
    parquet_path = hf_hub_download(repo_id="witgaw/METR-LA", repo_type="dataset", filename=f"{split_name}.parquet")
    df = pd.read_parquet(parquet_path, columns=["node_id", "t0_timestamp", *_metrla_feature_columns()])
    df = df.sort_values(["t0_timestamp", "node_id"], kind="mergesort")
    timestamps = df["t0_timestamp"].drop_duplicates().tolist()[: int(max_windows)]
    feature_cols = _metrla_feature_columns()
    samples: list[np.ndarray] = []
    for timestamp in timestamps:
        group = df[df["t0_timestamp"] == timestamp].sort_values("node_id", kind="mergesort")
        if len(group) == 0:
            continue
        node_values = group[feature_cols].to_numpy(dtype=np.float32)
        if node_values.shape[0] == 0:
            continue
        seq_len = len(feature_cols) // 2
        sample = node_values.reshape(node_values.shape[0], seq_len, 2).transpose(1, 0, 2)
        samples.append(sample)
    if not samples:
        raise ValueError(f"no windows could be built from METR-LA split {split_name!r}.")
    return np.stack(samples, axis=0).astype(np.float32), {"split": split_name, "parquet_path": parquet_path, "windows": len(samples)}


@lru_cache(maxsize=8)
def _load_metrla_hf_splits_cached(train_samples: int, val_samples: int, test_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    train_x, train_meta = _load_metrla_hf_split("train", train_samples)
    val_x, val_meta = _load_metrla_hf_split("val", val_samples)
    test_x, test_meta = _load_metrla_hf_split("test", test_samples)
    adj_path = hf_hub_download(repo_id="witgaw/METR-LA", repo_type="dataset", filename="sensor_graph/adj_mx.npy")
    adj = np.asarray(np.load(adj_path, allow_pickle=True), dtype=np.float32)
    degree = adj.sum(axis=1, keepdims=True)
    adj = adj / np.clip(degree, 1.0, None)
    metadata = {
        "dataset_name": "METR-LA",
        "fallback_used": False,
        "source": "huggingface",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "test_meta": test_meta,
        "adjacency_path": adj_path,
        "adjacency_shape": list(adj.shape),
    }
    return train_x, val_x, test_x, adj, metadata


def build_dataset_bundle(config: dict) -> DatasetBundle:
    dataset_cfg = config["dataset"]
    seed = int(config.get("seed", 1))
    name = dataset_cfg["name"]

    real_bundle = None
    if name == "pems08_debug":
        real_bundle = _load_pems08_debug_splits(dataset_cfg, seed)

    if real_bundle is not None:
        train_x, val_x, test_x, adj, metadata = real_bundle
        train_x, val_x, test_x, scaler = _normalize_splits(train_x, val_x, test_x)
    elif name in {"toy", "pems08_debug"}:
        train_x, adj = _generate_toy_tensor(
            num_samples=int(dataset_cfg["train_samples"]),
            seq_len=int(dataset_cfg["seq_len"]),
            num_nodes=int(dataset_cfg["nodes"]),
            channels=int(dataset_cfg["channels"]),
            seed=seed,
        )
        val_x, _ = _generate_toy_tensor(
            num_samples=int(dataset_cfg["val_samples"]),
            seq_len=int(dataset_cfg["seq_len"]),
            num_nodes=int(dataset_cfg["nodes"]),
            channels=int(dataset_cfg["channels"]),
            seed=seed + 1,
        )
        test_x, _ = _generate_toy_tensor(
            num_samples=int(dataset_cfg["test_samples"]),
            seq_len=int(dataset_cfg["seq_len"]),
            num_nodes=int(dataset_cfg["nodes"]),
            channels=int(dataset_cfg["channels"]),
            seed=seed + 2,
        )
        train_x, val_x, test_x, scaler = _normalize_splits(train_x, val_x, test_x)
        metadata = {
            "dataset_name": name,
            "fallback_used": name != "toy",
            "fallback_reason": "real PEMS08 data file not found" if name == "pems08_debug" else None,
        }
    else:
        raise ValueError(f"Unsupported dataset name for stage 1: {name}")

    missing_rate = float(dataset_cfg["missing_rate"])
    train_mask = random_missing_mask(train_x.shape, missing_rate, seed=seed)
    val_mask = random_missing_mask(val_x.shape, missing_rate, seed=seed + 11)
    test_mask = random_missing_mask(test_x.shape, missing_rate, seed=seed + 29)

    train_obs = train_x * train_mask
    val_obs = val_x * val_mask
    test_obs = test_x * test_mask

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"].get("num_workers", 0))

    train_loader = DataLoader(
        TrafficWindowDataset(train_x, train_obs, train_mask),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        TrafficWindowDataset(val_x, val_obs, val_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        TrafficWindowDataset(test_x, test_obs, test_mask),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        adjacency=torch.tensor(adj, dtype=torch.float32),
        scaler=scaler,
        metadata=metadata,
    )


def build_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, DataLoader, torch.Tensor, StandardScaler, dict]:
    bundle = build_dataset_bundle(config)
    return bundle.train_loader, bundle.val_loader, bundle.test_loader, bundle.adjacency, bundle.scaler, bundle.metadata
