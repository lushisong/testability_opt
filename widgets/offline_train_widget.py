# -*- coding: utf-8 -*-
from __future__ import annotations

import os
from typing import Optional, Callable
from PyQt5 import QtWidgets, QtCore
import numpy as np

from core.data_io import random_dataset, Dataset
from experiments.train_offline import train_guided_offline, train_mip_offline


class OfflineTrainWidget(QtWidgets.QWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, dataset_provider: Optional[Callable] = None):
        super().__init__(parent)
        self.dataset_provider = dataset_provider

        # 数据来源
        self.chk_use_current = QtWidgets.QCheckBox("使用当前数据集页中的数据")
        self.chk_use_current.setChecked(True)

        self.spin_m = QtWidgets.QSpinBox()
        self.spin_m.setRange(2, 5000)
        self.spin_m.setValue(40)
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(2, 5000)
        self.spin_n.setValue(80)
        self.spin_density = QtWidgets.QDoubleSpinBox()
        self.spin_density.setRange(0.01, 0.99)
        self.spin_density.setSingleStep(0.01)
        self.spin_density.setValue(0.30)

        # 训练目标选择
        self.cmb_model = QtWidgets.QComboBox()
        self.cmb_model.addItems(["NN-Guided_Offline", "NN-MIP_Offline"])

        # Guided 参数
        self.spin_samples = QtWidgets.QSpinBox()
        self.spin_samples.setRange(10, 20000)
        self.spin_samples.setValue(200)
        self.spin_epochs_g = QtWidgets.QSpinBox()
        self.spin_epochs_g.setRange(1, 20000)
        self.spin_epochs_g.setValue(300)
        self.spin_hidden_g = QtWidgets.QSpinBox()
        self.spin_hidden_g.setRange(2, 1024)
        self.spin_hidden_g.setValue(32)
        self.spin_seed_g = QtWidgets.QSpinBox()
        self.spin_seed_g.setRange(0, 1_000_000)
        self.spin_seed_g.setValue(0)

        # MIP 参数
        self.spin_epochs_m = QtWidgets.QSpinBox()
        self.spin_epochs_m.setRange(1, 20000)
        self.spin_epochs_m.setValue(200)
        self.spin_hidden_m = QtWidgets.QSpinBox()
        self.spin_hidden_m.setRange(2, 1024)
        self.spin_hidden_m.setValue(32)
        self.spin_seed_m = QtWidgets.QSpinBox()
        self.spin_seed_m.setRange(0, 1_000_000)
        self.spin_seed_m.setValue(0)

        # 输出路径
        self.txt_out = QtWidgets.QLineEdit(os.path.join(os.getcwd(), "data", "models", "offline_model.npz"))
        self.btn_browse = QtWidgets.QPushButton("浏览…")
        self.btn_browse.clicked.connect(self.on_browse)

        # 控件布局
        form = QtWidgets.QFormLayout()
        form.addRow(self.chk_use_current)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("m:"))
        size_row.addWidget(self.spin_m)
        size_row.addWidget(QtWidgets.QLabel("n:"))
        size_row.addWidget(self.spin_n)
        size_row.addWidget(QtWidgets.QLabel("density:"))
        size_row.addWidget(self.spin_density)
        form.addRow("随机数据规模：", size_row)
        form.addRow("模型：", self.cmb_model)

        # Guided 参数分组
        guided_row = QtWidgets.QHBoxLayout()
        guided_row.addWidget(QtWidgets.QLabel("samples:"))
        guided_row.addWidget(self.spin_samples)
        guided_row.addWidget(QtWidgets.QLabel("epochs:"))
        guided_row.addWidget(self.spin_epochs_g)
        guided_row.addWidget(QtWidgets.QLabel("hidden:"))
        guided_row.addWidget(self.spin_hidden_g)
        guided_row.addWidget(QtWidgets.QLabel("seed:"))
        guided_row.addWidget(self.spin_seed_g)
        form.addRow("Guided 训练参数：", guided_row)

        # MIP 参数分组
        mip_row = QtWidgets.QHBoxLayout()
        mip_row.addWidget(QtWidgets.QLabel("epochs:"))
        mip_row.addWidget(self.spin_epochs_m)
        mip_row.addWidget(QtWidgets.QLabel("hidden:"))
        mip_row.addWidget(self.spin_hidden_m)
        mip_row.addWidget(QtWidgets.QLabel("seed:"))
        mip_row.addWidget(self.spin_seed_m)
        form.addRow("MIP 训练参数：", mip_row)

        # 输出
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.txt_out)
        out_row.addWidget(self.btn_browse)
        form.addRow("输出模型路径：", out_row)

        # 训练按钮与进度
        self.btn_start = QtWidgets.QPushButton("开始训练")
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        self.progress.setValue(0)
        self.lbl_status = QtWidgets.QLabel("等待开始…")

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(self.btn_start)
        layout.addWidget(self.progress)
        layout.addWidget(self.lbl_status)

        self.btn_start.clicked.connect(self.on_start)
        self.cmb_model.currentTextChanged.connect(self.on_model_change)
        self.on_model_change(self.cmb_model.currentText())

    def on_browse(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存模型", self.txt_out.text(), "NPZ (*.npz)")
        if path:
            self.txt_out.setText(path)

    def _get_dataset(self) -> Dataset:
        if self.chk_use_current.isChecked() and self.dataset_provider:
            ds = self.dataset_provider()
            if ds is not None:
                return ds
        # 否则生成随机
        m = int(self.spin_m.value())
        n = int(self.spin_n.value())
        dens = float(self.spin_density.value())
        return random_dataset(m, n, density=dens, seed=0)

    def on_model_change(self, name: str):
        # 简单的提示：不同模型关注不同参数
        if name == "NN-Guided_Offline":
            self.spin_samples.setEnabled(True)
            self.spin_epochs_g.setEnabled(True)
            self.spin_hidden_g.setEnabled(True)
            self.spin_seed_g.setEnabled(True)
            self.spin_epochs_m.setEnabled(False)
            self.spin_hidden_m.setEnabled(False)
            self.spin_seed_m.setEnabled(False)
        else:
            self.spin_samples.setEnabled(False)
            self.spin_epochs_g.setEnabled(False)
            self.spin_hidden_g.setEnabled(False)
            self.spin_seed_g.setEnabled(False)
            self.spin_epochs_m.setEnabled(True)
            self.spin_hidden_m.setEnabled(True)
            self.spin_seed_m.setEnabled(True)

    def on_start(self):
        ds = self._get_dataset()
        out_path = self.txt_out.text().strip()
        if not out_path:
            QtWidgets.QMessageBox.warning(self, "提示", "请设置输出模型路径")
            return
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        model = self.cmb_model.currentText()
        self.progress.setValue(5)
        QtWidgets.QApplication.processEvents()
        try:
            if model == "NN-Guided_Offline":
                train_guided_offline(
                    ds.D, ds.fault_probs, ds.test_costs, out_path,
                    synth_samples=int(self.spin_samples.value()),
                    epochs=int(self.spin_epochs_g.value()),
                    hidden=int(self.spin_hidden_g.value()),
                    seed=int(self.spin_seed_g.value()),
                )
            else:
                train_mip_offline(
                    ds.D, ds.fault_probs, ds.test_costs, out_path,
                    epochs=int(self.spin_epochs_m.value()),
                    hidden=int(self.spin_hidden_m.value()),
                    seed=int(self.spin_seed_m.value()),
                )
            self.progress.setValue(100)
            self.lbl_status.setText(f"训练完成：{out_path}")
            QtWidgets.QMessageBox.information(self, "完成", f"模型已保存到：\n{out_path}")
        except Exception as e:
            self.lbl_status.setText(f"训练失败：{e}")
            QtWidgets.QMessageBox.critical(self, "错误", f"训练失败：\n{e}")

