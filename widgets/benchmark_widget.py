# -*- coding: utf-8 -*-
import os
from typing import Callable, Optional
from PyQt5 import QtWidgets
from core.benchmark import run_benchmark, summarize_and_plot
from widgets.mpl_canvas import SimplePlot

class BenchmarkWidget(QtWidgets.QWidget):
    def __init__(self, parent=None, dataset_provider: Optional[Callable]=None):
        super().__init__(parent)
        self.dataset_provider = dataset_provider

        # 选择算法
        self.chk_greedy = QtWidgets.QCheckBox("Greedy")
        self.chk_greedy.setChecked(True)
        self.chk_firefly = QtWidgets.QCheckBox("Firefly")
        self.chk_firefly.setChecked(True)
        self.chk_pso = QtWidgets.QCheckBox("BinaryPSO")
        self.chk_pso.setChecked(True)
        self.chk_nn = QtWidgets.QCheckBox("NN-Guided")
        self.chk_nn.setChecked(True)
        self.chk_nnmip = QtWidgets.QCheckBox("NN-MIP")
        self.chk_nnmip.setChecked(False)
        self.chk_nn_off = QtWidgets.QCheckBox("NN-Guided_Offline")
        self.chk_nn_off.setChecked(False)
        self.chk_nnmip_off = QtWidgets.QCheckBox("NN-MIP_Offline")
        self.chk_nnmip_off.setChecked(False)

        self.spin_repeats = QtWidgets.QSpinBox()
        self.spin_repeats.setRange(1, 100)
        self.spin_repeats.setValue(8)

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
        self.spin_budget.setValue(0.0)

        self.btn_run = QtWidgets.QPushButton("运行 Benchmark")
        self.lbl_csv = QtWidgets.QLabel("结果 CSV：尚未生成")

        # 图片预览
        self.plot_view = SimplePlot()

        # Layout
        form = QtWidgets.QFormLayout()
        algo_row = QtWidgets.QHBoxLayout()
        algo_row.addWidget(self.chk_greedy)
        algo_row.addWidget(self.chk_firefly)
        algo_row.addWidget(self.chk_pso)
        algo_row.addWidget(self.chk_nn)
        algo_row.addWidget(self.chk_nnmip)
        algo_row.addWidget(self.chk_nn_off)
        algo_row.addWidget(self.chk_nnmip_off)
        form.addRow("选择算法：", algo_row)
        form.addRow("重复次数：", self.spin_repeats)
        form.addRow("FDR 阈值 τ_d：", self.spin_tau_d)
        form.addRow("FIR 阈值 τ_i：", self.spin_tau_i)
        form.addRow("总成本上限(0=不限)：", self.spin_budget)
        form.addRow(self.btn_run)
        form.addRow(self.lbl_csv)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.plot_view)

        self.btn_run.clicked.connect(self.on_run)

    def on_run(self):
        ds = self.dataset_provider() if self.dataset_provider else None
        if ds is None:
            QtWidgets.QMessageBox.warning(self, "提示", "请先在数据集页生成或打开数据。")
            return

        algos = []
        if self.chk_greedy.isChecked(): algos.append("Greedy")
        if self.chk_firefly.isChecked(): algos.append("Firefly")
        if self.chk_pso.isChecked(): algos.append("BinaryPSO")
        if self.chk_nn.isChecked(): algos.append("NN-Guided")
        if self.chk_nnmip.isChecked(): algos.append("NN-MIP")
        if self.chk_nn_off.isChecked(): algos.append("NN-Guided_Offline")
        if self.chk_nnmip_off.isChecked(): algos.append("NN-MIP_Offline")
        if not algos:
            QtWidgets.QMessageBox.information(self, "提示", "请至少选择一种算法。")
            return

        repeats = int(self.spin_repeats.value())
        tau_d = float(self.spin_tau_d.value())
        tau_i = float(self.spin_tau_i.value())

        budget = float(self.spin_budget.value())
        if budget <= 0.0:
            budget = None
        results = run_benchmark(ds.D, ds.fault_probs, ds.test_costs, tau_d, tau_i, algos, repeats=repeats, base_seed=123, budget=budget)
        out = summarize_and_plot(results, out_dir=os.path.join(os.getcwd(), "results"))
        self.lbl_csv.setText(f"结果 CSV：{out['csv']}")

        # 预览一张图（相对最优成本 CDF）
        preview_png = os.path.join(os.getcwd(), "results", "cdf_relative_cost.png")
        if os.path.exists(preview_png):
            self.plot_view.show_image_file(preview_png)
        else:
            # 备用：绘制简单条形图
            import numpy as np
            by_algo = {}
            for r in results:
                by_algo.setdefault(r["algo"], []).append(r["cost"])
            labels = list(by_algo.keys())
            vals = [float(np.mean(by_algo[k])) for k in labels]
            self.plot_view.plot_bar(labels, vals, "Mean Cost by Algorithm", "Cost")
