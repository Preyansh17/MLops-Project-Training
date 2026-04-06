
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset
from tqdm.auto import tqdm


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    with open(config_path, "r") as f:
        if config_path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        if config_path.suffix.lower() == ".json":
            return json.load(f)
        raise ValueError(f"Unsupported config format: {config_path.suffix}")


def ensure_dirs(*paths: Path):
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rankdata(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    obs = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense_rank = obs.cumsum() - 1
    counts = np.bincount(dense_rank)
    cumulative = np.cumsum(counts)
    starts = cumulative - counts
    avg_ranks = (starts + cumulative - 1) / 2.0 + 1.0
    return avg_ranks[dense_rank][inv]


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt((x ** 2).sum()) * np.sqrt((y ** 2).sum())
    if denom == 0:
        return float("nan")
    return float((x * y).sum() / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rankdata(x), rankdata(y))


class GlobalMLP(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dims: list[int] | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [512, 128, 32]
        dims = [input_dim] + hidden_dims
        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class GlobalEmbeddingDataset(Dataset):
    def __init__(self, split_df: pd.DataFrame, cache_dir: Path):
        self.df = split_df.reset_index(drop=True).copy()
        self.cache_dir = cache_dir

        self.embeddings: list[torch.Tensor] = []
        self.targets: list[torch.Tensor] = []
        self.image_paths: list[str] = []

        for row in self.df.itertuples(index=False):
            rel_path = row.image_name
            target = float(row.quality_mos_norm)
            split_name = row.split
            cache_path = embedding_cache_path(self.cache_dir, split_name, rel_path)
            emb = np.load(cache_path).astype(np.float32)

            self.embeddings.append(torch.from_numpy(emb))
            self.targets.append(torch.tensor([target], dtype=torch.float32))
            self.image_paths.append(rel_path)


    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "emb": self.embeddings[idx],
            "target": self.targets[idx],
            "image_path": self.image_paths[idx],
        }


def collate_global(batch):
    return {
        "emb": torch.stack([b["emb"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "image_path": [b["image_path"] for b in batch],
    }


@torch.no_grad()
def evaluate_global(model, loader, device):
    model.eval()
    mse_vals, mae_vals, preds, targets, paths = [], [], [], [], []

    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["emb"].to(device).float()
        y = batch["target"].to(device).float()
        pred = model(x)
        mse_vals.append(F.mse_loss(pred, y).item())
        mae_vals.append(F.l1_loss(pred, y).item())
        preds.append(pred.squeeze(1).detach().cpu().numpy())
        targets.append(y.squeeze(1).detach().cpu().numpy())
        paths.extend(batch["image_path"])

    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])

    metrics = {
        "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
        "mae": float(np.mean(mae_vals)) if mae_vals else float("nan"),
        "plcc": float(pearson_corr(preds, targets)) if len(preds) else float("nan"),
        "srcc": float(spearman_corr(preds, targets)) if len(preds) else float("nan"),
    }
    pred_df = pd.DataFrame({"image_path": paths, "target_quality_mos_norm": targets, "predicted_score": preds})
    return metrics, pred_df


def train_one_epoch_global(model, loader, optimizer, device):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["emb"].to(device).float()
        y = batch["target"].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def embedding_cache_path(cache_dir: Path, split_name: str, image_rel_path: str) -> Path:
    safe_name = image_rel_path.replace("/", "__").replace("\\", "__")
    out_dir = cache_dir / split_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{safe_name}.npy"
