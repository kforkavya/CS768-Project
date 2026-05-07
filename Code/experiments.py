"""
experiments.py
==============
Training loop, evaluation, memory tracking, result saving (CSV / JSON),
and reproduction of all paper figures and tables.

Quick-start (demo on ogbn-arxiv, 5 epochs):
    python experiments.py --dataset arxiv --demo

Full experiment on ogbn-proteins (reproduces paper Table 1):
    python experiments.py --dataset proteins --layers 1001 --model revgnn

All results are saved to ./results/<run_name>/ as CSV files.
All plots are saved to ./plots/.
"""

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch_geometric.loader import ClusterData, ClusterLoader, NeighborLoader
from tqdm import tqdm

from revgnn import (
    ResGNN, RevGNN, DEQGNN,
    build_model, print_complexity_table,
    reset_peak_memory, peak_memory_gb, current_memory_gb,
    theoretical_memory,
)

from torch_geometric.data.data import Data, DataEdgeAttr

torch.serialization.add_safe_globals([
    Data,
    DataEdgeAttr
])
# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_proteins(root: str = "./data"):
    """Load ogbn-proteins. Returns (data, split_idx, evaluator)."""
    dataset = PygNodePropPredDataset("ogbn-proteins", root=root)
    data = dataset[0]
    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbn-proteins")

    # ogbn-proteins has no node features; initialise them from edge features
    # by sum-aggregating the 8-dim edge attributes (Section B.2 in the paper)
    from torch_scatter import scatter
    row, col = data.edge_index
    data.x = scatter(
        data.edge_attr, col, dim=0,
        dim_size=data.num_nodes, reduce="sum"
    )
    return data, split_idx, evaluator, "proteins"


def load_arxiv(root: str = "./data"):
    """Load ogbn-arxiv. Returns (data, split_idx, evaluator)."""
    dataset = PygNodePropPredDataset("ogbn-arxiv", root=root)
    data = dataset[0]

    # Convert directed → undirected (Section B.2)
    from torch_geometric.transforms import ToUndirected, AddSelfLoops
    data = ToUndirected()(data)
    data = AddSelfLoops()(data)

    split_idx = dataset.get_idx_split()
    evaluator = Evaluator(name="ogbn-arxiv")
    return data, split_idx, evaluator, "arxiv"


# ─────────────────────────────────────────────────────────────────────────────
# Mini-batch helpers (ClusterGCN-style random partitioning)
# ─────────────────────────────────────────────────────────────────────────────

def make_cluster_loaders(data, num_parts_train=10, num_parts_val=5, batch_size=1):
    """Random-clustering mini-batch loaders (Section 3.4 in the paper)."""
    train_cluster = ClusterData(data, num_parts=num_parts_train, recursive=False)
    train_loader = ClusterLoader(train_cluster, batch_size=batch_size, shuffle=True)

    val_cluster = ClusterData(data, num_parts=num_parts_val, recursive=False)
    val_loader = ClusterLoader(val_cluster, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, task="binary"):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        edge_attr = batch.edge_attr if hasattr(batch, "edge_attr") else None
        out = model(batch.x, batch.edge_index, edge_attr)

        if task == "binary":
            mask = batch.train_mask if hasattr(batch, "train_mask") else torch.ones(out.size(0), dtype=torch.bool)
            y = batch.y[mask].float()
            loss = F.binary_cross_entropy_with_logits(out[mask], y)
        else:
            mask = batch.train_mask if hasattr(batch, "train_mask") else torch.ones(out.size(0), dtype=torch.bool)
            y = batch.y[mask].squeeze()
            loss = F.cross_entropy(out[mask], y)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# @torch.no_grad()
# def evaluate(model, data, split_idx, evaluator, device, task="binary"):
#     model.eval()
#     x = data.x.to(device)
#     edge_index = data.edge_index.to(device)
#     edge_attr = data.edge_attr.to(device) if hasattr(data, "edge_attr") and data.edge_attr is not None else None

#     out = model(x, edge_index, edge_attr)

#     results = {}
#     for split, idx in split_idx.items():
#         y_pred = out[idx]
#         y_true = data.y[idx]
#         results[split] = evaluator.eval({
#             "y_pred": y_pred,
#             "y_true": y_true,
#         })[evaluator.eval_metric]

#     return results

@torch.no_grad()
def evaluate(model, data, split_idx, evaluator, device, task="binary"):
    model.eval()

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = (
        data.edge_attr.to(device)
        if hasattr(data, "edge_attr") and data.edge_attr is not None
        else None
    )

    out = model(x, edge_index, edge_attr)

    results = {}
    for split, idx in split_idx.items():

        if task == "multiclass":
            # Convert logits -> predicted class ids
            y_pred = out[idx].argmax(dim=-1, keepdim=True)
        else:
            # proteins case: keep raw logits
            y_pred = out[idx]

        y_true = data.y[idx]

        results[split] = evaluator.eval({
            "y_true": y_true,
            "y_pred": y_pred,
        })[evaluator.eval_metric]

    return results


def train_model(
    model_type: str,
    hidden_channels: int,
    num_layers: int,
    dataset_name: str,
    epochs: int = 2000,
    lr: float = 1e-3,
    dropout: float = 0.1,
    conv_type: str = "gen",
    num_parts_train: int = 10,
    num_parts_val: int = 5,
    num_trials: int = 3,
    device: str = "auto",
    results_dir: str = "./results",
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train a model and return a DataFrame with results.

    Saves intermediate results to CSV so training can be interrupted/resumed.
    """
    set_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
          if device == "auto" else torch.device(device)

    # Load dataset
    if dataset_name == "proteins":
        data, split_idx, evaluator, ds = load_proteins()
        task = "binary"
        in_channels, out_channels, edge_dim = 8, 112, 8
    elif dataset_name == "arxiv":
        data, split_idx, evaluator, ds = load_arxiv()
        task = "multiclass"
        in_channels = data.x.shape[1]
        out_channels = int(data.y.max()) + 1
        edge_dim = None
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    run_name = f"{model_type}_{hidden_channels}ch_{num_layers}L_{dataset_name}"
    run_dir = Path(results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "metrics.csv"

    all_results = []

    for trial in range(num_trials):
        set_seed(seed + trial)

        # Build model
        model = build_model(
            model_type, in_channels, hidden_channels, out_channels,
            num_layers, dropout, conv_type, edge_dim,
        ).to(dev)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.5)

        # Data loaders
        train_loader, val_loader = make_cluster_loaders(
            data, num_parts_train, num_parts_val
        )

        # Measure peak GPU memory on first epoch
        reset_peak_memory()

        trial_rows = []
        best_val = 0.0

        for epoch in tqdm(range(1, epochs + 1), desc=f"Trial {trial+1}/{num_trials}", disable=not verbose):
            t0 = time.time()
            loss = train_epoch(model, train_loader, optimizer, dev, task)
            scheduler.step()

            if epoch % 10 == 0 or epoch == 1:
                evals = evaluate(model, data, split_idx, evaluator, dev, task)
                mem_gb = peak_memory_gb()

                row = {
                    "trial": trial,
                    "epoch": epoch,
                    "loss": round(loss, 6),
                    "train_score": round(evals.get("train", 0), 6),
                    "val_score": round(evals.get("valid", 0), 6),
                    "test_score": round(evals.get("test", 0), 6),
                    "peak_mem_gb": round(mem_gb, 4),
                    "model_type": model_type,
                    "hidden_channels": hidden_channels,
                    "num_layers": num_layers,
                    "n_params": model.count_parameters(),
                    "elapsed_s": round(time.time() - t0, 2),
                }
                trial_rows.append(row)

                if evals.get("valid", 0) > best_val:
                    best_val = evals["valid"]

                if verbose and epoch % 50 == 0:
                    print(
                        f"  Trial {trial+1} | Epoch {epoch:4d} | "
                        f"Loss {loss:.4f} | Val {evals.get('valid',0):.4f} | "
                        f"Mem {mem_gb:.2f} GB"
                    )

                # Save incrementally
                pd.DataFrame(trial_rows).to_csv(
                    run_dir / f"trial_{trial}_metrics.csv", index=False
                )

        all_results.extend(trial_rows)
        df_all = pd.DataFrame(all_results)
        df_all.to_csv(csv_path, index=False)

        if verbose:
            print(
                f"Trial {trial+1}/{num_trials} finished | "
                f"Best val: {best_val:.4f} | "
                f"Peak mem: {peak_memory_gb():.2f} GB | "
                f"Params: {model.count_parameters():,}"
            )

    return pd.DataFrame(all_results)


# ─────────────────────────────────────────────────────────────────────────────
# Quick memory benchmark (no full training needed)
# ─────────────────────────────────────────────────────────────────────────────

def memory_benchmark(
    model_types: List[str],
    hidden_channels_list: List[int],
    num_layers_list: List[int],
    dataset_name: str = "proteins",
    device: str = "auto",
    results_dir: str = "./results",
    num_fwd_passes: int = 3,
) -> pd.DataFrame:
    """
    Measure ACTUAL peak GPU memory for different models / depths
    without running a full training loop.

    Reproduces the data behind Figures 2-4 in the paper.
    """
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
          if device == "auto" else torch.device(device)

    if dataset_name == "proteins":
        data, split_idx, _, ds = load_proteins()
        in_channels, out_channels, edge_dim = 8, 112, 8
        task = "binary"
        conv_type = "gen"
    else:
        data, split_idx, _, ds = load_arxiv()
        in_channels = data.x.shape[1]
        out_channels = int(data.y.max()) + 1
        edge_dim = None
        task = "multiclass"
        conv_type = "gcn"

    # Use a small subgraph for quick benchmarking
    cluster = ClusterData(data, num_parts=10, recursive=False)
    from torch_geometric.loader import ClusterLoader
    loader = ClusterLoader(cluster, batch_size=1, shuffle=False)
    batch = next(iter(loader)).to(dev)
    edge_attr = batch.edge_attr.to(dev) if hasattr(batch, "edge_attr") and batch.edge_attr is not None else None

    rows = []
    for model_type in model_types:
        for hidden in hidden_channels_list:
            for n_layers in num_layers_list:
                tag = f"{model_type}_{hidden}ch_{n_layers}L"
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc_collect()
                reset_peak_memory()

                try:
                    model = build_model(
                        model_type, in_channels, hidden, out_channels,
                        n_layers, 0.1, conv_type, edge_dim,
                    ).to(dev).train()

                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                    peak = 0.0
                    for _ in range(num_fwd_passes):
                        optimizer.zero_grad()
                        out = model(batch.x, batch.edge_index, edge_attr)
                        if task == "binary":
                            loss = F.binary_cross_entropy_with_logits(out, batch.y.float())
                        else:
                            loss = F.cross_entropy(out, batch.y.squeeze())
                        loss.backward()
                        optimizer.step()
                        peak = max(peak, peak_memory_gb())
                        reset_peak_memory()

                    n_params = model.count_parameters()
                    oom = False

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        peak = float("nan")
                        n_params = -1
                        oom = True
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    else:
                        raise

                row = {
                    "model_type": model_type,
                    "hidden_channels": hidden,
                    "num_layers": n_layers,
                    "peak_mem_gb": round(peak, 4),
                    "n_params": n_params,
                    "oom": oom,
                    "dataset": dataset_name,
                }
                rows.append(row)
                status = "OOM" if oom else f"{peak:.3f} GB"
                print(f"  {tag:<30} mem={status:<10} params={n_params:,}" if not oom
                      else f"  {tag:<30} OUT OF MEMORY")

                # Free model
                del model
                gc_collect()

    df = pd.DataFrame(rows)
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{results_dir}/memory_benchmark_{dataset_name}.csv", index=False)
    return df


def gc_collect():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Plotting (reproduces paper figures)
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "resgnn":    "#e74c3c",   # red
    "revgnn":    "#2ecc71",   # green
    "wt-resgnn": "#e67e22",   # orange
    "wt-revgnn": "#9b59b6",   # purple
    "deq":       "#3498db",   # blue
}

LABELS = {
    "resgnn":    "ResGNN (baseline)",
    "revgnn":    "RevGNN (ours)",
    "wt-resgnn": "WT-ResGNN",
    "wt-revgnn": "WT-RevGNN",
    "deq":       "DEQ-GNN",
}

LINE_STYLE = {
    "resgnn":    "-",
    "revgnn":    "-",
    "wt-resgnn": "--",
    "wt-revgnn": "--",
    "deq":       ":",
}


def plot_memory_vs_layers(
    df: pd.DataFrame,
    title: str = "GPU Memory vs. Number of Layers",
    save_path: str = "./plots/memory_vs_layers.png",
):
    """
    Reproduce Figure 2 in the paper:
    GPU memory consumption vs. number of layers for ResGNN and RevGNN.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for mt, grp in df.groupby("model_type"):
        grp_sorted = grp.sort_values("num_layers")
        color = PALETTE.get(mt, "grey")
        ls = LINE_STYLE.get(mt, "-")
        label = LABELS.get(mt, mt)

        # OOM points
        oom = grp_sorted[grp_sorted["oom"] == True]
        valid = grp_sorted[grp_sorted["oom"] == False]

        if not valid.empty:
            ax.plot(
                valid["num_layers"], valid["peak_mem_gb"],
                color=color, linestyle=ls, linewidth=2,
                marker="o", markersize=7, label=label,
                zorder=3,
            )
            # Annotate with score if available
            if "val_score" in valid.columns:
                for _, r in valid.iterrows():
                    ax.annotate(
                        f"{r['val_score']:.2f}",
                        (r["num_layers"], r["peak_mem_gb"]),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=7, color=color,
                    )

        # Mark OOM
        for _, r in oom.iterrows():
            ax.annotate(
                "OOM", (r["num_layers"], 34),
                ha="center", color=color, fontsize=8, fontweight="bold",
            )

    ax.axhline(y=11, color="grey", linestyle="--", alpha=0.6, linewidth=1.2, label="11 GB GPU")
    ax.axhline(y=32, color="black", linestyle="--", alpha=0.4, linewidth=1.2, label="32 GB GPU")

    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Peak GPU Memory (GB)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 36)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_score_vs_memory(
    records: List[Dict],
    title: str = "ROC-AUC vs. GPU Memory (Figure 1)",
    save_path: str = "./plots/score_vs_memory.png",
):
    """
    Reproduce Figure 1 in the paper:
    ROC-AUC score vs. GPU memory consumption, bubble size ∝ √params.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))

    for r in records:
        n_params = r.get("n_params", 1e6)
        bubble = max(30, 80 * (n_params / 1e7) ** 0.5)
        color = PALETTE.get(r["model_type"], "grey")
        label = r.get("label", LABELS.get(r["model_type"], r["model_type"]))

        if r.get("oom", False):
            ax.scatter(r["mem_gb"] + 2, r["score"],
                       s=bubble, color=color, marker="x", linewidths=2, alpha=0.7)
            ax.annotate("OOM", (r["mem_gb"] + 2, r["score"]),
                        xytext=(5, 0), textcoords="offset points", fontsize=7, color=color)
        else:
            ax.scatter(r["mem_gb"], r["score"],
                       s=bubble, color=color, alpha=0.85, edgecolors="white", linewidth=0.5,
                       zorder=3)
            ax.annotate(label, (r["mem_gb"], r["score"]),
                        xytext=(6, 3), textcoords="offset points", fontsize=8, color=color)

    # Legend patches
    patches = [mpatches.Patch(color=v, label=LABELS.get(k, k)) for k, v in PALETTE.items()]
    ax.legend(handles=patches, fontsize=8, loc="lower right")

    ax.axvline(x=11, color="grey", linestyle="--", alpha=0.5, linewidth=1)
    ax.axvline(x=32, color="black", linestyle="--", alpha=0.3, linewidth=1)

    ax.set_xlabel("← GPU Memory (GB)", fontsize=12)
    ax.set_ylabel("Score (ROC-AUC) →", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_training_curves(
    df: pd.DataFrame,
    metric: str = "val_score",
    title: str = "Validation Score vs. Epoch",
    save_path: str = "./plots/training_curves.png",
):
    """Plot training curves grouped by model type."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))

    for mt, grp in df.groupby("model_type"):
        # Aggregate over trials
        agg = grp.groupby("epoch")[metric].agg(["mean", "std"])
        color = PALETTE.get(mt, "grey")
        label = LABELS.get(mt, mt)

        ax.plot(agg.index, agg["mean"], color=color, linewidth=2, label=label)
        ax.fill_between(
            agg.index,
            agg["mean"] - agg["std"],
            agg["mean"] + agg["std"],
            color=color, alpha=0.15,
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_theoretical_memory(
    model_types: List[str],
    layer_counts: List[int],
    N: int = 132_534,   # ogbn-proteins node count
    D: int = 80,
    save_path: str = "./plots/theoretical_memory.png",
):
    """
    Plot THEORETICAL memory complexity (not measured).
    Demonstrates O(ND) vs O(LND) scaling.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for mt in model_types:
        mems = [theoretical_memory(mt, L, N, D) for L in layer_counts]
        ax.plot(
            layer_counts, mems,
            color=PALETTE.get(mt, "grey"),
            linestyle=LINE_STYLE.get(mt, "-"),
            linewidth=2,
            label=LABELS.get(mt, mt),
            marker="o", markersize=4,
        )

    ax.set_xlabel("Number of Layers (L)", fontsize=12)
    ax.set_ylabel("Theoretical Activation Memory (GB)", fontsize=12)
    ax.set_title(
        f"Theoretical Memory: O(LND) vs O(ND)\n"
        f"N={N:,} nodes, D={D} channels",
        fontsize=12,
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


def plot_param_count(
    model_types: List[str],
    hidden_channels: int,
    layer_counts: List[int],
    in_channels: int = 8,
    out_channels: int = 112,
    edge_dim: int = 8,
    conv_type: str = "gcn",
    save_path: str = "./plots/param_count.png",
):
    """Show parameter counts as function of depth for different models."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    for mt in model_types:
        params = []
        for L in layer_counts:
            try:
                m = build_model(mt, in_channels, hidden_channels, out_channels,
                                L, 0.1, conv_type, edge_dim)
                params.append(m.count_parameters())
            except Exception:
                params.append(float("nan"))

        ax.plot(
            layer_counts, [p / 1e6 for p in params],
            color=PALETTE.get(mt, "grey"),
            linestyle=LINE_STYLE.get(mt, "-"),
            linewidth=2,
            label=LABELS.get(mt, mt),
            marker="o", markersize=4,
        )

    ax.set_xlabel("Number of Layers (L)", fontsize=12)
    ax.set_ylabel("Parameters (M)", fontsize=12)
    ax.set_title(
        f"Parameter Count vs. Depth\n(hidden={hidden_channels})", fontsize=12
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Pre-computed result tables (for fast re-plotting without re-training)
# ─────────────────────────────────────────────────────────────────────────────

PAPER_PROTEINS_RESULTS = [
    # Fig 1 data (Table 1 + Figure 1 in paper)
    {"label": "GCN",            "model_type": "resgnn",    "score": 72.51, "mem_gb":  4.68, "n_params":  96_900, "oom": False},
    {"label": "GraphSAGE",      "model_type": "resgnn",    "score": 77.68, "mem_gb":  3.12, "n_params": 193_000, "oom": False},
    {"label": "DeeperGCN",      "model_type": "resgnn",    "score": 86.16, "mem_gb": 27.1,  "n_params": 2_370_000, "oom": False},
    {"label": "GAT",            "model_type": "resgnn",    "score": 86.82, "mem_gb":  6.74, "n_params": 2_480_000, "oom": False},
    {"label": "UniMP+CEF",      "model_type": "resgnn",    "score": 86.91, "mem_gb": 27.2,  "n_params": 1_960_000, "oom": False},
    {"label": "RevGNN-Deep",    "model_type": "revgnn",    "score": 87.74, "mem_gb":  2.86, "n_params": 20_030_000, "oom": False},
    {"label": "RevGNN-Wide",    "model_type": "revgnn",    "score": 88.24, "mem_gb":  7.91, "n_params": 68_470_000, "oom": False},
]

# Data for Figure 2 (memory vs layers, from Table 6 and Figure 2 in paper)
PAPER_FIG2_DATA = pd.DataFrame([
    # ResGNN-64 (baseline)
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers":   3, "peak_mem_gb":  0.5, "val_score": 83.47, "oom": False},
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers":   7, "peak_mem_gb":  1.0, "val_score": 84.65, "oom": False},
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers":  14, "peak_mem_gb":  2.0, "val_score": 85.16, "oom": False},
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers":  28, "peak_mem_gb":  4.0, "val_score": 85.26, "oom": False},
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers":  56, "peak_mem_gb":  8.0, "val_score": 86.05, "oom": False},
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers": 112, "peak_mem_gb": 27.1, "val_score": 85.94, "oom": False},
    {"model_type": "resgnn", "hidden_channels": 64, "num_layers": 224, "peak_mem_gb": float("nan"), "val_score": float("nan"), "oom": True},
    # RevGNN-80
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers":   3, "peak_mem_gb":  0.6, "val_score": 84.71, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers":   7, "peak_mem_gb":  0.6, "val_score": 84.96, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers":  14, "peak_mem_gb":  0.6, "val_score": 85.31, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers":  28, "peak_mem_gb":  0.6, "val_score": 85.96, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers":  56, "peak_mem_gb":  0.6, "val_score": 85.97, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers": 112, "peak_mem_gb":  2.56, "val_score": 86.23, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers": 448, "peak_mem_gb":  2.56, "val_score": 86.83, "oom": False},
    {"model_type": "revgnn", "hidden_channels": 80, "num_layers":1001, "peak_mem_gb":  2.86, "val_score": 87.06, "oom": False},
    # RevGNN-224
    {"model_type": "revgnn", "hidden_channels":224, "num_layers":   3, "peak_mem_gb":  0.9, "val_score": 85.09, "oom": False},
    {"model_type": "revgnn", "hidden_channels":224, "num_layers":   7, "peak_mem_gb":  0.9, "val_score": 85.68, "oom": False},
    {"model_type": "revgnn", "hidden_channels":224, "num_layers":  14, "peak_mem_gb":  0.9, "val_score": 86.62, "oom": False},
    {"model_type": "revgnn", "hidden_channels":224, "num_layers":  28, "peak_mem_gb":  0.9, "val_score": 86.68, "oom": False},
    {"model_type": "revgnn", "hidden_channels":224, "num_layers":  56, "peak_mem_gb":  0.9, "val_score": 86.90, "oom": False},
    {"model_type": "revgnn", "hidden_channels":224, "num_layers": 112, "peak_mem_gb":  7.3,  "val_score": 87.02, "oom": False},
    {"model_type": "revgnn", "hidden_channels":224, "num_layers": 448, "peak_mem_gb":  7.3,  "val_score": 87.33, "oom": False},
])

# Table 6 (112-layer comparison on ogbn-proteins)
PAPER_TABLE6 = pd.DataFrame([
    {"model": "ResGNN-64",     "channels": 64,  "roc_auc": 85.94, "mem_gb": 27.1,  "n_params": 2_370_000, "time_days": 1.3},
    {"model": "ResGNN-224",    "channels": 224, "roc_auc": float("nan"), "mem_gb": float("nan"), "n_params": 28_400_000, "time_days": float("nan")},  # OOM
    {"model": "WT-ResGNN-64",  "channels": 64,  "roc_auc": 83.30, "mem_gb": 27.4,  "n_params": 51_200,    "time_days": 1.2},
    {"model": "DEQ-GNN-64",    "channels": 64,  "roc_auc": 83.17, "mem_gb":  2.22, "n_params": 51_300,    "time_days": 1.3},
    {"model": "DEQ-GNN-224",   "channels": 224, "roc_auc": 85.84, "mem_gb":  7.60, "n_params": 537_000,   "time_days": 2.9},
    {"model": "RevGNN-64",     "channels": 64,  "roc_auc": 85.48, "mem_gb":  2.09, "n_params": 1_460_000, "time_days": 1.8},
    {"model": "RevGNN-80",     "channels": 80,  "roc_auc": 85.97, "mem_gb":  2.56, "n_params": 2_250_000, "time_days": 2.2},
    {"model": "RevGNN-224",    "channels": 224, "roc_auc": 87.02, "mem_gb":  7.30, "n_params": 17_100_000,"time_days": 4.9},
    {"model": "WT-RevGNN-64",  "channels": 64,  "roc_auc": 82.89, "mem_gb":  1.60, "n_params": 35_000,    "time_days": 1.7},
    {"model": "WT-RevGNN-80",  "channels": 80,  "roc_auc": 83.46, "mem_gb":  2.08, "n_params": 51_400,    "time_days": 2.0},
    {"model": "WT-RevGNN-224", "channels": 224, "roc_auc": 85.28, "mem_gb":  5.55, "n_params": 337_000,   "time_days": 4.8},
])


def plot_paper_figures(save_dir: str = "./plots"):
    """
    Reproduce all paper figures using the pre-extracted paper numbers.
    Call this even without training to get publication-quality plots.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    print("\n[Plots] Generating paper figures from pre-extracted results ...")

    # ── Figure 1: Score vs Memory (bubble chart) ─────────────────────────────
    plot_score_vs_memory(
        PAPER_PROTEINS_RESULTS,
        title="Figure 1: ROC-AUC vs GPU Memory on ogbn-proteins\n(bubble size ∝ √params)",
        save_path=f"{save_dir}/fig1_score_vs_memory.png",
    )

    # ── Figure 2: Memory vs Layers ────────────────────────────────────────────
    fig2_subset = PAPER_FIG2_DATA[
        PAPER_FIG2_DATA["model_type"].isin(["resgnn", "revgnn"])
    ].copy()
    plot_memory_vs_layers(
        fig2_subset,
        title="Figure 2: GPU Memory vs. Layers for ResGNN and RevGNN\n(ogbn-proteins)",
        save_path=f"{save_dir}/fig2_memory_vs_layers.png",
    )

    # ── Figure 3: Weight-tied variants ───────────────────────────────────────
    fig3_data = pd.DataFrame([
        {"model_type": "wt-resgnn", "num_layers":  3, "peak_mem_gb":  0.9, "val_score": 82.76, "oom": False},
        {"model_type": "wt-resgnn", "num_layers":  7, "peak_mem_gb":  1.8, "val_score": 83.41, "oom": False},
        {"model_type": "wt-resgnn", "num_layers": 14, "peak_mem_gb":  3.6, "val_score": 83.67, "oom": False},
        {"model_type": "wt-resgnn", "num_layers": 28, "peak_mem_gb":  7.0, "val_score": 83.35, "oom": False},
        {"model_type": "wt-resgnn", "num_layers": 56, "peak_mem_gb": 14.0, "val_score": 82.91, "oom": False},
        {"model_type": "wt-resgnn", "num_layers":112, "peak_mem_gb": 27.4, "val_score": 83.30, "oom": False},
        {"model_type": "wt-revgnn", "num_layers":  3, "peak_mem_gb":  0.6, "val_score": 82.55, "oom": False},
        {"model_type": "wt-revgnn", "num_layers":  7, "peak_mem_gb":  0.6, "val_score": 83.28, "oom": False},
        {"model_type": "wt-revgnn", "num_layers": 14, "peak_mem_gb":  0.6, "val_score": 83.53, "oom": False},
        {"model_type": "wt-revgnn", "num_layers": 28, "peak_mem_gb":  0.6, "val_score": 83.10, "oom": False},
        {"model_type": "wt-revgnn", "num_layers": 56, "peak_mem_gb":  0.6, "val_score": 83.07, "oom": False},
        {"model_type": "wt-revgnn", "num_layers":112, "peak_mem_gb":  5.55,"val_score": 85.28, "oom": False},
    ])
    plot_memory_vs_layers(
        fig3_data,
        title="Figure 3: GPU Memory vs. Layers – Weight-Tied Variants",
        save_path=f"{save_dir}/fig3_weight_tied_memory.png",
    )

    # ── Figure 4: DEQ variants ────────────────────────────────────────────────
    fig4_data = pd.DataFrame([
        {"model_type": "wt-resgnn", "num_layers":  3, "peak_mem_gb":  0.9, "val_score": 82.76, "oom": False},
        {"model_type": "wt-resgnn", "num_layers": 14, "peak_mem_gb":  3.6, "val_score": 83.67, "oom": False},
        {"model_type": "wt-resgnn", "num_layers": 56, "peak_mem_gb": 14.0, "val_score": 82.91, "oom": False},
        {"model_type": "wt-resgnn", "num_layers":112, "peak_mem_gb": 27.4, "val_score": 83.30, "oom": False},
        {"model_type": "deq",       "num_layers":  3, "peak_mem_gb":  2.2, "val_score": 79.04, "oom": False},
        {"model_type": "deq",       "num_layers":  7, "peak_mem_gb":  2.2, "val_score": 82.82, "oom": False},
        {"model_type": "deq",       "num_layers": 14, "peak_mem_gb":  2.2, "val_score": 82.88, "oom": False},
        {"model_type": "deq",       "num_layers": 56, "peak_mem_gb":  2.2, "val_score": 83.17, "oom": False},
        {"model_type": "deq",       "num_layers":112, "peak_mem_gb":  7.6, "val_score": 85.84, "oom": False},
    ])
    plot_memory_vs_layers(
        fig4_data,
        title="Figure 4: GPU Memory vs. Iterations – DEQ-GNN",
        save_path=f"{save_dir}/fig4_deq_memory.png",
    )

    # ── Theoretical memory scaling ─────────────────────────────────────────────
    plot_theoretical_memory(
        model_types=["resgnn", "revgnn", "wt-resgnn", "wt-revgnn", "deq"],
        layer_counts=[1, 3, 7, 14, 28, 56, 112, 224, 448, 1001],
        save_path=f"{save_dir}/theoretical_memory_scaling.png",
    )

    # ── Parameter count vs depth ───────────────────────────────────────────────
    plot_param_count(
        model_types=["resgnn", "revgnn", "wt-resgnn", "wt-revgnn"],
        hidden_channels=80,
        layer_counts=[3, 7, 14, 28, 56, 112],
        save_path=f"{save_dir}/param_count_vs_depth.png",
    )

    # ── Table 6 (bar chart) ────────────────────────────────────────────────────
    _plot_table6(save_dir)

    print("[Plots] Done. All figures saved to:", save_dir)


def _plot_table6(save_dir: str):
    """Plot Table 6 as a grouped bar chart."""
    df = PAPER_TABLE6.dropna(subset=["roc_auc"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Table 6: 112-layer Model Comparison on ogbn-proteins", fontsize=13)

    colors = plt.cm.Set2(range(len(df)))

    for ax, col, ylabel in zip(
        axes,
        ["roc_auc", "mem_gb", "n_params"],
        ["ROC-AUC (%)", "Peak GPU Memory (GB)", "Parameters"],
    ):
        bars = ax.bar(df["model"], df[col], color=colors, edgecolor="white")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(df["model"], rotation=45, ha="right", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        if col == "n_params":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M")
            )

        for bar, val in zip(bars, df[col]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.01,
                f"{val:.2f}" if col != "n_params" else f"{val/1e6:.1f}M",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()
    plt.savefig(f"{save_dir}/table6_bar_chart.png", dpi=150)
    plt.close()
    print(f"  Saved → {save_dir}/table6_bar_chart.png")


def print_paper_tables():
    """Print Tables 1, 2, 6 from the paper."""
    print("\n" + "=" * 65)
    print("Table 1: Results on ogbn-proteins (ROC-AUC %)")
    print("=" * 65)
    t1 = pd.DataFrame(PAPER_PROTEINS_RESULTS)[["label", "score", "mem_gb", "n_params"]]
    t1.columns = ["Model", "ROC-AUC ↑", "Mem (GB) ↓", "Params"]
    t1["Params"] = t1["Params"].apply(lambda x: f"{x/1e6:.2f}M")
    print(t1.to_string(index=False))

    print("\n" + "=" * 65)
    print("Table 6: 112-layer comparison on ogbn-proteins")
    print("=" * 65)
    print(PAPER_TABLE6.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="RevGNN: Training Graph Neural Networks with 1000 Layers"
    )
    parser.add_argument("--dataset", default="arxiv",
                        choices=["arxiv", "proteins"],
                        help="Dataset to use")
    parser.add_argument("--model", default="revgnn",
                        choices=["resgnn", "revgnn", "wt-resgnn", "wt-revgnn", "deq"],
                        help="Model type")
    parser.add_argument("--layers", type=int, default=28,
                        help="Number of GNN layers")
    parser.add_argument("--hidden", type=int, default=64,
                        help="Hidden channel size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs")
    parser.add_argument("--trials", type=int, default=1,
                        help="Number of independent trials")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--results_dir", default="./results")
    parser.add_argument("--plots_dir", default="./plots")
    parser.add_argument("--demo", action="store_true",
                        help="Run a quick demo with minimal settings")
    parser.add_argument("--plots_only", action="store_true",
                        help="Only generate plots from pre-computed paper data")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run memory benchmark across model types and depths")
    args = parser.parse_args()

    print_complexity_table()
    print_paper_tables()

    if args.plots_only:
        plot_paper_figures(args.plots_dir)
        return

    if args.demo:
        args.epochs = 5
        args.layers = 7
        args.hidden = 32
        args.trials = 1
        print(f"\n[Demo mode] Quick run: {args.layers} layers, {args.epochs} epochs")

    if args.benchmark:
        print("\n[Benchmark] Running memory benchmark ...")
        df = memory_benchmark(
            model_types=["resgnn", "revgnn", "wt-resgnn", "wt-revgnn"],
            hidden_channels_list=[64],
            num_layers_list=[3, 7, 14, 28, 56, 112],
            dataset_name=args.dataset,
            results_dir=args.results_dir,
        )
        print(df.to_string())
        plot_memory_vs_layers(df, save_path=f"{args.plots_dir}/benchmark_memory.png")
        return

    print(f"\n[Training] {args.model} | {args.layers}L | {args.hidden}ch | "
          f"{args.dataset} | {args.epochs} epochs")

    df = train_model(
        model_type=args.model,
        hidden_channels=args.hidden,
        num_layers=args.layers,
        dataset_name=args.dataset,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        num_trials=args.trials,
        results_dir=args.results_dir,
        verbose=True,
    )

    plot_training_curves(
        df,
        save_path=f"{args.plots_dir}/training_{args.model}_{args.layers}L_{args.hidden}ch_{args.epochs}epochs_{args.dataset}.png"
    )
    # plot_paper_figures(args.plots_dir)

    print("\n[Done] Results summary:")
    last_epoch = df.groupby("trial")["epoch"].max()
    for trial, ep in last_epoch.items():
        row = df[(df["trial"] == trial) & (df["epoch"] == ep)].iloc[0]
        print(
            f"  Trial {trial+1}: val={row['val_score']:.4f}  "
            f"test={row['test_score']:.4f}  "
            f"mem={row['peak_mem_gb']:.2f}GB  "
            f"params={row['n_params']:,}"
        )


if __name__ == "__main__":
    main()
