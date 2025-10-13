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
        self.cmb_algo.addItems(["Greedy", "Firefly", "BinaryPSO", "NN-Guided"])

        self.spin_tau_d = QtWidgets.QDoubleSpinBox()
        self.spin_tau_d.setRange(0.0, 1.0)
        self.spin_tau_d.setSingleStep(0.01)
        self.spin_tau_d.setValue(0.9)
        self.spin_tau_i = QtWidgets.QDoubleSpinBox()
        self.spin_tau_i.setRange(0.0, 1.0)
        self.spin_tau_i.setSingleStep(0.01)
        self.spin_tau_i.setValue(0.8)

        self.btn_run = QtWidgets.QPushButton("运行")
        self.text_out = QtWidgets.QPlainTextEdit()
        self.text_out.setReadOnly(True)

        form = QtWidgets.QFormLayout()
        form.addRow("算法：", self.cmb_algo)
        form.addRow("FDR 阈值 τ_d：", self.spin_tau_d)
        form.addRow("FIR 阈值 τ_i：", self.spin_tau_i)
        form.addRow(self.btn_run)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.text_out)

        self.btn_run.clicked.connect(self.on_run)

    def on_run(self):
        ds = self.dataset_provider() if self.dataset_provider else None
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在数据集页生成或打开数据。")
            return
        algo_name = self.cmb_algo.currentText()
        tau_d = float(self.spin_tau_d.value())
        tau_i = float(self.spin_tau_i.value())

        if algo_name == "Greedy":
            algo = GreedyAlgo()
            res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42)
        elif algo_name == "Firefly":
            algo = FireflyAlgo()
            res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42)
        elif algo_name == "BinaryPSO":
            algo = BinaryPSOAlgo()
            res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42)
        else:
            algo = NNGuidedAlgo()
            res = algo.run(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, seed=42)

        sel_idx = np.where(res.selected == 1)[0]
        sel_names = [ds.test_names[i] for i in sel_idx]
        lines = []
        lines.append(f"算法：{res.name}")
        lines.append(f"FDR={res.fdr:.4f}, FIR={res.fir:.4f}, Cost={res.cost:.4f}, 选中数={len(sel_idx)}, Time={res.runtime_sec:.4f}s")
        lines.append(f"选中测试：{', '.join(sel_names)}")
        self.text_out.setPlainText("\n".join(lines))
