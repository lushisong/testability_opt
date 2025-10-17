import numpy as np
import pytest

from core.algos.neural_branching import NeuralBranchingAlgo
from core.algos.utils import BinaryMetricHelper
from core.solvers import BranchingStrategy, solve_tp_mip_cp_sat
from experiments.branching_data import BranchingDataset, BranchingSample
from experiments.models import BranchingGCN, save_branching_gcn


def test_branching_dataset_roundtrip(tmp_path):
    feats = np.array([[1.0, 2.0], [3.0, 4.0]])
    adj = np.eye(2)
    selected = np.array([0, 1], dtype=np.uint8)
    candidate = np.array([1, 0], dtype=np.uint8)
    sample = BranchingSample(
        features=feats,
        adjacency=adj,
        selected_mask=selected,
        candidate_mask=candidate,
        best_index=0,
    )
    dataset = BranchingDataset([sample])
    path = tmp_path / "data.npz"
    dataset.save_npz(path)
    loaded = BranchingDataset.load_npz(path)
    assert len(loaded.samples) == 1
    s = loaded.samples[0]
    assert s.features.shape == (2, 2)
    assert s.adjacency.shape == (2, 2)
    assert s.best_index == 0


def test_cp_sat_branching_strategy_integration():
    D = np.array([[1, 0, 1], [0, 1, 1]], dtype=int)
    probs = np.array([0.6, 0.4])
    costs = np.array([1.0, 2.0, 0.5])
    order = [2, 0, 1]
    strategy = BranchingStrategy(order=order, var_strategy="CHOOSE_FIRST", domain_strategy="SELECT_MIN_VALUE")
    sol = solve_tp_mip_cp_sat(
        D,
        probs,
        costs,
        tau_d=0.1,
        tau_i=0.1,
        time_limit_s=0.1,
        branching=strategy,
        num_workers=1,
    )
    assert sol["branching_strategy"]["order"][:3] == order


def test_branching_gcn_forward_shape():
    torch = pytest.importorskip("torch")
    model = BranchingGCN(in_dim=5, hidden_dim=4, num_layers=1, use_mask_feature=True)
    feats = torch.rand(4, 5)
    adj = torch.eye(4)
    mask = torch.zeros(4)
    logits = model(feats, adj, mask)
    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_neural_branching_learned_and_default(tmp_path):
    torch = pytest.importorskip("torch")
    D = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=int)
    probs = np.array([0.3, 0.4, 0.3])
    costs = np.array([1.0, 2.0, 3.0])
    helper = BinaryMetricHelper(D, probs, costs)
    feats = helper.feature_matrix(np.zeros(helper.n, dtype=np.uint8))
    model = BranchingGCN(in_dim=feats.shape[1], hidden_dim=4, num_layers=1, use_mask_feature=True)
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.convs[0].linear.weight[0, 1] = 1.0
        model.head.weight[0, 0] = -1.0
    model_path = tmp_path / "model.pt"
    save_branching_gcn(
        model_path,
        model,
        meta={
            "in_dim": feats.shape[1],
            "hidden_dim": 4,
            "layers": 1,
            "use_mask_feature": True,
        },
    )
    algo = NeuralBranchingAlgo(model_path=str(model_path))
    res_default = algo.run(
        D,
        probs,
        costs,
        tau_d=0.1,
        tau_i=0.1,
        time_limit=0.1,
        branching_mode="default",
        num_workers=1,
    )
    assert res_default.extra["branching_mode"] == "default"
    res_learned = algo.run(
        D,
        probs,
        costs,
        tau_d=0.1,
        tau_i=0.1,
        time_limit=0.1,
        branching_mode="learned",
        num_workers=1,
    )
    assert res_learned.extra["branching_mode"] == "learned"
    assert res_learned.extra["logits"] is not None
