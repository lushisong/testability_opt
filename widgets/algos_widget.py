# -*- coding: utf-8 -*-
from typing import Callable, Optional
from PyQt5 import QtWidgets
import numpy as np

from core.metrics import fdr, fir, cost
from core.algos.greedy import GreedyAlgo
from core.algos.firefly import FireflyAlgo
from core.algos.pso import BinaryPSOAlgo
from core.algos.nn_guided import NNGuidedAlgo

class AlgosWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, dataset_provider: Optional[Callable]=None):
        super().__init__(parent)
        self.dataset_provider = dataset_provider

        self.cmb_algo = QtWidgets.QComboBox()
        self.cmb_algo.addItems(["Greedy", "Firefly", "BinaryPSO", "NN-Guided", "NN-MIP", "NN-Guided_Offline", "NN-MIP_Offline"])

        self.spin_tau_d = QtWidgets.QDoubleSpinBox()
        self.spin_tau_d.setRange(0.0, 1.0)
        self.spin_tau_d.setSingleStep(0.01)
        self.spin_tau_d.setValue(0.9)
        self.spin_tau_i = QtWidgets.QDoubleSpinBox()
        self.spin_tau_i.setRange(0.0, 1.0)
        self.spin_tau_i.setSingleStep(0.01)
        self.spin_tau_i.setValue(0.8)

        self.spin_budget = QtWidgets.QDoubleSpinBox()
        self.spin_budget.setRange(0.0, 1e9)
        self.spin_budget.setDecimals(3)
        self.spin_budget.setSingleStep(1.0)
        self.spin_budget.setValue(0.0)  # 0 表示不限

        # 离线模型路径
        self.txt_model_path = QtWidgets.QLineEdit()
        self.txt_model_path.setPlaceholderText("如 data/models/xxx_offline.npz")
        self.btn_model_browse = QtWidgets.QPushButton("浏览…")
        self.btn_model_browse.clicked.connect(self.on_browse_model)

        self.btn_run = QtWidgets.QPushButton("运行")
        self.text_out = QtWidgets.QPlainTextEdit()
        self.text_out.setReadOnly(True)

        form = QtWidgets.QFormLayout()
        form.addRow("算法：", self.cmb_algo)
        form.addRow("FDR 阈值 τ_d：", self.spin_tau_d)
        form.addRow("FIR 阈值 τ_i：", self.spin_tau_i)
        form.addRow("总成本上限(0=不限)：", self.spin_budget)
        row_model = QtWidgets.QHBoxLayout()
        row_model.addWidget(self.txt_model_path)
        row_model.addWidget(self.btn_model_browse)
        form.addRow("离线模型路径(.npz)：", row_model)
        form.addRow(self.btn_run)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.text_out)

        self.btn_run.clicked.connect(self.on_run)
        self.cmb_algo.currentTextChanged.connect(self.on_algo_change)
        self.on_algo_change(self.cmb_algo.currentText())

    def on_browse_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择模型文件", self.txt_model_path.text(), "NPZ (*.npz)")
        if path:
            self.txt_model_path.setText(path)

    def on_algo_change(self, name: str):
        is_offline = name in ("NN-Guided_Offline", "NN-MIP_Offline")
        self.txt_model_path.setEnabled(is_offline)
        self.btn_model_browse.setEnabled(is_offline)

    def on_run(self):
        ds = self.dataset_provider() if self.dataset_provider else None
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在数据集页生成或打开数据。")
            return
        algo_name = self.cmb_algo.currentText()
        tau_d = float(self.spin_tau_d.value())
        tau_i = float(self.spin_tau_i.value())
        budget = float(self.spin_budget.value())
        if budget <= 0.0:
            budget = None

        try:
            if algo_name == "Greedy":
                algo = GreedyAlgo()
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42, budget=budget)
            elif algo_name == "Firefly":
                algo = FireflyAlgo()
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42, budget=budget)
            elif algo_name == "BinaryPSO":
                algo = BinaryPSOAlgo()
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42, budget=budget)
            elif algo_name == "NN-MIP":
                from core.algos.nn_mip import NNHintMIPAlgo
                algo = NNHintMIPAlgo()
                # UI 环境下避免多线程与回调引发原生崩溃：使用单线程且不注册回调
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42,
                               budget=budget, num_workers=1, use_callback=False)
            elif algo_name == "NN-Guided_Offline":
                from core.algos.nn_guided_offline import NNGuidedOfflineAlgo
                algo = NNGuidedOfflineAlgo()
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42, budget=budget,
                               model_path=self.txt_model_path.text().strip() or None)
            elif algo_name == "NN-MIP_Offline":
                from core.algos.nn_mip_offline import NNHintMIPOfflineAlgo
                algo = NNHintMIPOfflineAlgo()
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42,
                               budget=budget, num_workers=1, use_callback=False,
                               model_path=self.txt_model_path.text().strip() or None)
            else:
                algo = NNGuidedAlgo()
                res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42, budget=budget)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "算法运行错误", f"{algo_name} 执行失败：\n{e}")
            return

        sel_idx = np.where(res.selected == 1)[0]
        sel_names = [ds.test_names[i] for i in sel_idx]
        lines = []
        lines.append(f"算法：{res.name}")
        lines.append(f"FDR={res.fdr:.4f}, FIR={res.fir:.4f}, Cost={res.cost:.4f}, 选中数={len(sel_idx)}, Time={res.runtime_sec:.4f}s")
        lines.append(f"选中测试：{', '.join(sel_names)}")
        self.text_out.setPlainText("\n".join(lines))
