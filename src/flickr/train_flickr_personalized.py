from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import mlflow
import pandas as pd
import torch
from torch.utils.data import DataLoader

from flickr_common import (
    PersonalizedEmbeddingDataset,
    PersonalizedMLP,
    collate_personalized,
    ensure_dirs,
    evaluate_personalized,
    load_config,
    set_seed,
    train_one_epoch_personalized,
)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


def safe_log_params(cfg: dict):
    flat_cfg = flatten_dict(cfg)
    safe_params = {}
    for k, v in flat_cfg.items():
        if isinstance(v, (str, int, float, bool)) or v is None:
            safe_params[k] = v
        else:
            safe_params[k] = str(v)
    mlflow.log_params(safe_params)


def main(config_path: str):
    cfg = load_config(config_path)
    set_seed(cfg["seed"])

    paths = cfg["paths"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    mlflow_cfg = cfg.get("mlflow", {})

    experiment_name = mlflow_cfg.get("experiment_name", "flickr-personalized")
    run_name = mlflow_cfg.get("run_name", paths.get("run_name", "personalized_run"))

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    output_root = Path(paths["output_root"])
    run_dir = output_root / paths.get("run_name", "personalized_run")
    ckpt_dir = run_dir / "checkpoints"
    pred_dir = run_dir / "predictions"
    ensure_dirs(run_dir, ckpt_dir, pred_dir)

    manifest = pd.read_csv(paths["manifest_csv"])
    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    val_df = manifest[manifest["split"] == "val"].reset_index(drop=True)
    test_df = manifest[manifest["split"] == "test"].reset_index(drop=True)
    production_seen_df = manifest[manifest["split"] == "production_seen"].reset_index(drop=True)
    production_new_user_df = manifest[manifest["split"] == "production_new_user"].reset_index(drop=True)

    cache_dir = output_root / paths["cache_subdir"]
    device = train_cfg["device"]

    all_seen_workers = sorted(
        manifest.loc[manifest["worker_split"] == "seen_worker_pool", "worker_id"].unique()
    )
    user2idx = {u: i for i, u in enumerate(all_seen_workers)}
    num_users = len(user2idx)

    train_ds = PersonalizedEmbeddingDataset(train_df, cache_dir, user2idx)
    val_ds = PersonalizedEmbeddingDataset(val_df, cache_dir, user2idx)
    test_ds = PersonalizedEmbeddingDataset(test_df, cache_dir, user2idx)
    prod_seen_ds = PersonalizedEmbeddingDataset(production_seen_df, cache_dir, user2idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_personalized,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_personalized,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_personalized,
    )
    prod_seen_loader = DataLoader(
        prod_seen_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_personalized,
    )

    model = PersonalizedMLP(
        num_users=num_users,
        input_dim=model_cfg["input_dim"],
        user_emb_dim=model_cfg["user_emb_dim"],
        hidden_dims=model_cfg["hidden_dims"],
        dropout=model_cfg["dropout"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    history = []
    best_val_srcc = -1e9
    best_ckpt = ckpt_dir / "best_personalized_model.pth"
    last_ckpt = ckpt_dir / "last_personalized_model.pth"

    try: 
        mlflow.end_run() 
    except:
        pass
    finally:
        mlflow.start_run(run_name = run_name,log_system_metrics=True)

    mlflow.set_tags(
        {
            "model_type": "personalized",
            "dataset": "FLICKR-AES",
            "task": "personalized_aesthetic_prediction",
            "framework": "pytorch",
        }
    )

    safe_log_params(cfg)

    mlflow.log_metrics(
        {
            "num_train_rows": len(train_df),
            "num_val_rows": len(val_df),
            "num_test_rows": len(test_df),
            "num_production_seen_rows": len(production_seen_df),
            "num_production_new_user_rows": len(production_new_user_df),
            "num_seen_workers_total": len(all_seen_workers),
            "num_train_workers": train_df["worker_id"].nunique(),
            "num_val_workers": val_df["worker_id"].nunique(),
            "num_test_workers": test_df["worker_id"].nunique(),
            "num_production_seen_workers": production_seen_df["worker_id"].nunique(),
            "num_production_new_user_workers": production_new_user_df["worker_id"].nunique(),
        },
        step=0,
    )

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        epoch_start = time.time()

        train_mse = train_one_epoch_personalized(model, train_loader, optimizer, device)
        val_metrics, _ = evaluate_personalized(model, val_loader, device)

        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mse": val_metrics["mse"],
            "val_mae": val_metrics["mae"],
            "val_plcc": val_metrics["plcc"],
            "val_srcc": val_metrics["srcc"],
            "epoch_time_sec": epoch_time,
        }
        history.append(row)
        print(row)

        mlflow.log_metrics(
            {
                "train_mse": train_mse,
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "val_plcc": val_metrics["plcc"],
                "val_srcc": val_metrics["srcc"],
                "epoch_time_sec": epoch_time,
            },
            step=epoch,
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            last_ckpt,
        )

        if val_metrics["srcc"] > best_val_srcc:
            best_val_srcc = val_metrics["srcc"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                },
                best_ckpt,
            )
            mlflow.log_metric("best_val_srcc", best_val_srcc, step=epoch)

    best_bundle = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(best_bundle["model_state_dict"])

    val_metrics, val_pred = evaluate_personalized(model, val_loader, device)
    test_metrics, test_pred = evaluate_personalized(model, test_loader, device)
    prod_seen_metrics, prod_seen_pred = evaluate_personalized(model, prod_seen_loader, device)

    history_csv = run_dir / "history.csv"
    val_pred_csv = pred_dir / "val_predictions.csv"
    test_pred_csv = pred_dir / "test_predictions.csv"
    prod_seen_pred_csv = pred_dir / "production_seen_predictions.csv"
    metrics_csv = run_dir / "metrics.csv"

    pd.DataFrame(history).to_csv(history_csv, index=False)
    val_pred.to_csv(val_pred_csv, index=False)
    test_pred.to_csv(test_pred_csv, index=False)
    prod_seen_pred.to_csv(prod_seen_pred_csv, index=False)

    pd.DataFrame(
        [
            {"split": "val", **val_metrics},
            {"split": "test", **test_metrics},
            {"split": "production_seen", **prod_seen_metrics},
        ]
    ).to_csv(metrics_csv, index=False)

    mlflow.log_metrics(
        {
            "final_val_mse": val_metrics["mse"],
            "final_val_mae": val_metrics["mae"],
            "final_val_plcc": val_metrics["plcc"],
            "final_val_srcc": val_metrics["srcc"],
            "test_mse": test_metrics["mse"],
            "test_mae": test_metrics["mae"],
            "test_plcc": test_metrics["plcc"],
            "test_srcc": test_metrics["srcc"],
            "production_seen_mse": prod_seen_metrics["mse"],
            "production_seen_mae": prod_seen_metrics["mae"],
            "production_seen_plcc": prod_seen_metrics["plcc"],
            "production_seen_srcc": prod_seen_metrics["srcc"],
        }
    )

    mlflow.log_artifact(str(best_ckpt))
    mlflow.log_artifact(str(last_ckpt))
    mlflow.log_artifact(str(history_csv))
    mlflow.log_artifact(str(metrics_csv))
    mlflow.log_artifact(str(val_pred_csv))
    mlflow.log_artifact(str(test_pred_csv))
    mlflow.log_artifact(str(prod_seen_pred_csv))

    mlflow.end_run()
    print("Saved outputs to", run_dir)


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/train_personalized.yaml"
    main(config_path)