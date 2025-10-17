# -*- coding: utf-8 -*-
"""Training script for the branching GCN model."""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

from experiments.branching_data import BranchingDataset
from experiments.models import BranchingGCN, save_branching_gcn


def _prepare_tensors(sample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    x = torch.from_numpy(sample.features.astype(np.float32))
    adj = torch.from_numpy(sample.adjacency.astype(np.float32))
    selected = torch.from_numpy(sample.selected_mask.astype(np.float32))
    best = int(sample.best_index)
    return x, adj, selected, best


def train_model(dataset: BranchingDataset, hidden_dim: int, layers: int, lr: float, epochs: int, device: str) -> BranchingGCN:
    if len(dataset.samples) == 0:
        raise ValueError("Branching dataset is empty")
    in_dim = dataset.samples[0].features.shape[1]
    model = BranchingGCN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=layers, use_mask_feature=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for sample in dataset.samples:
            feat, adj, selected, best = _prepare_tensors(sample)
            feat, adj, selected = feat.to(device), adj.to(device), selected.to(device)
            optimizer.zero_grad()
            logits = model(feat, adj, selected)
            candidates = torch.nonzero(
                torch.from_numpy(sample.candidate_mask.astype(np.uint8)), as_tuple=False
            ).squeeze(-1)
            if candidates.numel() == 0:
                continue
            cand_logits = logits[candidates.to(device)]
            target_idx = int((candidates == best).nonzero(as_tuple=True)[0])
            loss = F.cross_entropy(
                cand_logits.unsqueeze(0), torch.tensor([target_idx], dtype=torch.long, device=device)
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
        if (epoch + 1) % max(1, epochs // 5) == 0:
            avg_loss = total_loss / max(1, len(dataset.samples))
            print(f"[Epoch {epoch+1}] avg loss={avg_loss:.4f}")
    model.eval()
    return model


def run_inference(model: BranchingGCN, sample) -> np.ndarray:
    device = next(model.parameters()).device
    feat, adj, selected, _ = _prepare_tensors(sample)
    with torch.no_grad():
        logits = model(feat.to(device), adj.to(device), selected.to(device))
    return logits.cpu().numpy()


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Train a branching policy GCN model")
    parser.add_argument("data", type=str, help="Path to branching dataset (.npz)")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="branching_gcn.pt")
    parser.add_argument("--preview", action="store_true", help="Print logits of the first sample after training")
    args = parser.parse_args()

    dataset = BranchingDataset.load_npz(args.data)
    model = train_model(dataset, hidden_dim=args.hidden, layers=args.layers, lr=args.lr, epochs=args.epochs, device=args.device)
    save_branching_gcn(args.out, model, meta={"in_dim": dataset.samples[0].features.shape[1], "hidden_dim": args.hidden, "layers": args.layers, "use_mask_feature": True})

    if args.preview and dataset.samples:
        scores = run_inference(model, dataset.samples[0])
        print("Top logits:", scores)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
