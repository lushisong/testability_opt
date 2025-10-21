# -*- coding: utf-8 -*-
import json
import numpy as np

try:  # Optional dependency for GNN models
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests via skip
    torch = None
    _TORCH_IMPORT_ERROR = exc
else:
    _TORCH_IMPORT_ERROR = None

class TinyMLP:
    def __init__(self, in_dim, hidden=32, lr=1e-2, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.1, size=(in_dim, hidden))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, size=(hidden, 1))
        self.b2 = np.zeros(1)
        self.lr = lr

    def forward(self, X):
        self.X = X
        self.Hpre = X @ self.W1 + self.b1
        self.H = np.maximum(0.0, self.Hpre)
        self.Y = self.H @ self.W2 + self.b2
        return self.Y

    def backward(self, gradY):
        dW2 = self.H.T @ gradY
        db2 = gradY.sum(axis=0)
        dH = gradY @ self.W2.T
        dH[self.Hpre <= 0] = 0.0
        dW1 = self.X.T @ dH
        db1 = dH.sum(axis=0)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def fit_mse(self, X, y, epochs=200, batch=128):
        N = X.shape[0]
        for ep in range(epochs):
            idx = np.arange(N)
            np.random.shuffle(idx)
            for s in range(0, N, batch):
                part = idx[s:s+batch]
                pred = self.forward(X[part])
                err = pred - y[part].reshape(-1, 1)
                self.backward(2.0 * err / max(1, part.size))

    def predict(self, X):
        return self.forward(X).reshape(-1)


def save_tinymlp(path: str, net: TinyMLP, mu, sd, meta: dict | None = None):
    payload = {
        "W1": net.W1,
        "b1": net.b1,
        "W2": net.W2,
        "b2": net.b2,
        "mu": mu,
        "sd": sd,
    }
    if meta is not None:
        payload["meta_json"] = np.array(json.dumps(meta), dtype=np.str_)
    np.savez(path, **payload)


def load_tinymlp(path: str) -> tuple[TinyMLP, any, any, dict]:
    data = np.load(path, allow_pickle=False)
    W1, b1, W2, b2 = data["W1"], data["b1"], data["W2"], data["b2"]
    mu, sd = data["mu"], data["sd"]
    net = TinyMLP(in_dim=W1.shape[0], hidden=W1.shape[1])
    net.W1 = W1; net.b1 = b1; net.W2 = W2; net.b2 = b2
    meta_raw = data.get("meta_json", None)
    if meta_raw is None:
        meta = {}
    else:
        meta = json.loads(str(meta_raw))
    return net, mu, sd, meta


if torch is not None:

    class GraphConv(nn.Module):
        def __init__(self, in_dim: int, out_dim: int):
            super().__init__()
            self.linear = nn.Linear(in_dim, out_dim)

        def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
            device = x.device
            n = adj.size(0)
            eye = torch.eye(n, device=device, dtype=adj.dtype)
            adj_hat = adj + eye
            deg = adj_hat.sum(dim=-1)
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
            d_mat = torch.diag(deg_inv_sqrt)
            norm = d_mat @ adj_hat @ d_mat
            return self.linear(norm @ x)


    class BranchingGCN(nn.Module):
        """Simple GCN with optional mask feature for branching policy scoring."""

        def __init__(self, in_dim: int, hidden_dim: int = 32, num_layers: int = 2,
                     dropout: float = 0.0, use_mask_feature: bool = True):
            super().__init__()
            if num_layers < 1:
                raise ValueError("num_layers must be >= 1")
            self.use_mask_feature = bool(use_mask_feature)
            input_dim = in_dim + (1 if self.use_mask_feature else 0)
            self.convs = nn.ModuleList()
            last_dim = input_dim
            for _ in range(num_layers):
                self.convs.append(GraphConv(last_dim, hidden_dim))
                last_dim = hidden_dim
            self.head = nn.Linear(hidden_dim, 1)
            self.dropout = float(dropout)

        def forward(self, features: torch.Tensor, adj: torch.Tensor,
                    selected_mask: torch.Tensor | None = None) -> torch.Tensor:
            if self.use_mask_feature:
                if selected_mask is None:
                    raise ValueError("selected_mask must be provided when use_mask_feature=True")
                features = torch.cat([features, selected_mask.unsqueeze(-1)], dim=-1)
            x = features
            for conv in self.convs:
                x = conv(x, adj)
                x = F.relu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
            logits = self.head(x).squeeze(-1)
            return logits


    def save_branching_gcn(path: str, model: BranchingGCN, meta: dict | None = None) -> None:
        payload = {
            "state_dict": model.state_dict(),
            "meta": meta or {},
        }
        torch.save(payload, path)


    def load_branching_gcn(path: str, map_location: str | torch.device = "cpu") -> tuple[BranchingGCN, dict]:
        payload = torch.load(path, map_location=map_location)
        meta = payload.get("meta", {})
        model = BranchingGCN(
            in_dim=int(meta.get("in_dim", 5)),
            hidden_dim=int(meta.get("hidden_dim", 32)),
            num_layers=int(meta.get("layers", 2)),
            dropout=float(meta.get("dropout", 0.0)),
            use_mask_feature=bool(meta.get("use_mask_feature", True)),
        )
        model.load_state_dict(payload["state_dict"])
        model.to(map_location)
        model.eval()
        return model, meta

else:  # torch not available

    class BranchingGCN:  # type: ignore[misc]
        def __init__(self, *_, **__):
            raise RuntimeError("PyTorch is required for BranchingGCN") from _TORCH_IMPORT_ERROR


    def save_branching_gcn(*_, **__):  # pragma: no cover - requires torch
        raise RuntimeError("PyTorch is required to save BranchingGCN") from _TORCH_IMPORT_ERROR


    def load_branching_gcn(*_, **__):  # pragma: no cover - requires torch
        raise RuntimeError("PyTorch is required to load BranchingGCN") from _TORCH_IMPORT_ERROR
