from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix.lower() in {".yaml", ".yml"}:
            return yaml.safe_load(f)
        return json.load(f)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dirs(*paths: str | Path) -> None:
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def hash_path(path: str | Path) -> str:
    return hashlib.md5(str(Path(path).resolve()).encode("utf-8")).hexdigest()


COLAB_PREFIX = "/content/drive/MyDrive/FLICKR_AES"


def cache_path_for_image(cache_dir: str | Path, image_path: str | Path) -> Path:
    fake_abs_path = str(Path(COLAB_PREFIX) / str(image_path))
    
    key = fake_abs_path.encode("utf-8")
    name = hashlib.md5(key).hexdigest() + ".npy"
    
    return Path(cache_dir) / name


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
    def __init__(self, input_dim: int = 768, hidden_dims: List[int] | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [512, 128, 32]
        dims = [input_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class PersonalizedMLP(nn.Module):
    def __init__(self, num_users: int, input_dim: int = 768, user_emb_dim: int = 32, hidden_dims: List[int] | None = None, dropout: float = 0.1):
        super().__init__()
        hidden_dims = hidden_dims or [512, 128, 32]
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        dims = [input_dim + user_emb_dim] + hidden_dims
        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.extend([nn.Linear(dims[i], dims[i + 1]), nn.ReLU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, image_emb: torch.Tensor, user_idx: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_embedding(user_idx)
        x = torch.cat([image_emb, user_emb], dim=-1)
        return torch.sigmoid(self.net(x))


class GlobalEmbeddingDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, cache_dir: str | Path):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)

        emb_list = []
        target_list = []
        image_name_list = []
        split_list = []

        for _, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="Loading global embeddings",
        ):
            emb = np.load(
                cache_path_for_image(self.cache_dir, row["image_path"])
            ).astype(np.float32)

            emb_list.append(emb)
            target_list.append(float(row["global_score"]))
            image_name_list.append(row["image_name"])
            split_list.append(row["split"])

        self.embs = torch.tensor(np.stack(emb_list), dtype=torch.float32)
        self.targets = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1)
        self.image_names = image_name_list
        self.splits = split_list

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "emb": self.embs[idx],
            "target": self.targets[idx],
            "image_name": self.image_names[idx],
            "split": self.splits[idx],
        }


class PersonalizedEmbeddingDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, cache_dir: str | Path, user2idx: dict):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.cache_dir = Path(cache_dir)
        self.user2idx = user2idx

        emb_list = []
        user_idx_list = []
        target_list = []
        image_name_list = []
        worker_id_list = []
        split_list = []

        for _, row in tqdm(
            self.df.iterrows(),
            total=len(self.df),
            desc="Loading personalized embeddings",
        ):
            emb = np.load(
                cache_path_for_image(self.cache_dir, row["image_path"])
            ).astype(np.float32)

            emb_list.append(emb)
            user_idx_list.append(self.user2idx[row["worker_id"]])
            target_list.append(float(row["worker_score_norm"]))
            image_name_list.append(row["image_name"])
            worker_id_list.append(row["worker_id"])
            split_list.append(row["split"])

        self.embs = torch.tensor(np.stack(emb_list), dtype=torch.float32)
        self.user_idxs = torch.tensor(user_idx_list, dtype=torch.long)
        self.targets = torch.tensor(target_list, dtype=torch.float32).unsqueeze(1)
        self.image_names = image_name_list
        self.worker_ids = worker_id_list
        self.splits = split_list

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        return {
            "emb": self.embs[idx],
            "user_idx": self.user_idxs[idx],
            "target": self.targets[idx],
            "image_name": self.image_names[idx],
            "worker_id": self.worker_ids[idx],
            "split": self.splits[idx],
        }


def collate_global(batch):
    return {
        "emb": torch.stack([b["emb"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "image_name": [b["image_name"] for b in batch],
        "split": [b["split"] for b in batch],
    }


def collate_personalized(batch):
    return {
        "emb": torch.stack([b["emb"] for b in batch], dim=0),
        "user_idx": torch.stack([b["user_idx"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "image_name": [b["image_name"] for b in batch],
        "worker_id": [b["worker_id"] for b in batch],
        "split": [b["split"] for b in batch],
    }

def train_one_epoch_global(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
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


def train_one_epoch_personalized(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["emb"].to(device).float()
        u = batch["user_idx"].to(device)
        y = batch["target"].to(device).float()
        optimizer.zero_grad(set_to_none=True)
        pred = model(x, u)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_global(model: nn.Module, loader: DataLoader, device: str) -> Tuple[dict, pd.DataFrame]:
    model.eval()
    preds, targets, image_names = [], [], []
    mse_vals, mae_vals = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["emb"].to(device).float()
        y = batch["target"].to(device).float()
        pred = model(x)
        preds.append(pred.squeeze(1).cpu().numpy())
        targets.append(y.squeeze(1).cpu().numpy())
        image_names.extend(batch["image_name"])
        mse_vals.append(F.mse_loss(pred, y).item())
        mae_vals.append(F.l1_loss(pred, y).item())
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    metrics = {
        "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
        "mae": float(np.mean(mae_vals)) if mae_vals else float("nan"),
        "plcc": float(pearson_corr(preds, targets)) if len(preds) else float("nan"),
        "srcc": float(spearman_corr(preds, targets)) if len(preds) else float("nan"),
    }
    pred_df = pd.DataFrame({"image_name": image_names, "target": targets, "predicted_score": preds})
    return metrics, pred_df


@torch.no_grad()
def evaluate_personalized(model: nn.Module, loader: DataLoader, device: str) -> Tuple[dict, pd.DataFrame]:
    model.eval()
    preds, targets, image_names, worker_ids = [], [], [], []
    mse_vals, mae_vals = [], []
    for batch in tqdm(loader, desc="Eval", leave=False):
        x = batch["emb"].to(device).float()
        u = batch["user_idx"].to(device)
        y = batch["target"].to(device).float()
        pred = model(x, u)
        preds.append(pred.squeeze(1).cpu().numpy())
        targets.append(y.squeeze(1).cpu().numpy())
        image_names.extend(batch["image_name"])
        worker_ids.extend(batch["worker_id"])
        mse_vals.append(F.mse_loss(pred, y).item())
        mae_vals.append(F.l1_loss(pred, y).item())
    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    metrics = {
        "mse": float(np.mean(mse_vals)) if mse_vals else float("nan"),
        "mae": float(np.mean(mae_vals)) if mae_vals else float("nan"),
        "plcc": float(pearson_corr(preds, targets)) if len(preds) else float("nan"),
        "srcc": float(spearman_corr(preds, targets)) if len(preds) else float("nan"),
    }
    pred_df = pd.DataFrame({
        "worker_id": worker_ids,
        "image_name": image_names,
        "target": targets,
        "predicted_score": preds,
    })
    return metrics, pred_df
