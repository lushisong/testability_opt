# -*- coding: utf-8 -*-
import os
from typing import Optional
import numpy as np
from PyQt5 import QtWidgets, QtCore
from core.data_io import Dataset, random_dataset

class DatasetWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ds: Optional[Dataset] = None

        # Controls
        self.spin_m = QtWidgets.QSpinBox()
        self.spin_m.setRange(2, 500)
        self.spin_m.setValue(12)
        self.spin_n = QtWidgets.QSpinBox()
        self.spin_n.setRange(2, 500)
        self.spin_n.setValue(20)
        self.spin_density = QtWidgets.QDoubleSpinBox()
        self.spin_density.setRange(0.05, 0.95)
        self.spin_density.setSingleStep(0.05)
        self.spin_density.setValue(0.3)
        self.btn_gen = QtWidgets.QPushButton("随机生成 D 矩阵")
        self.btn_save = QtWidgets.QPushButton("保存为 JSON")
        self.btn_open = QtWidgets.QPushButton("打开 JSON")

        # Tables
        self.table_faults = QtWidgets.QTableWidget(0, 2)
        self.table_faults.setHorizontalHeaderLabels(["故障名称", "概率"])
        self.table_tests = QtWidgets.QTableWidget(0, 2)
        self.table_tests.setHorizontalHeaderLabels(["测试名称", "代价"])
        self.table_D = QtWidgets.QTableWidget(0, 0)

        # Layout
        form = QtWidgets.QFormLayout()
        form.addRow("故障数 m:", self.spin_m)
        form.addRow("测试数 n:", self.spin_n)
        form.addRow("覆盖密度:", self.spin_density)
        hl = QtWidgets.QHBoxLayout()
        hl.addWidget(self.btn_gen)
        hl.addWidget(self.btn_save)
        hl.addWidget(self.btn_open)
        form.addRow(hl)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        w1 = QtWidgets.QWidget()
        l1 = QtWidgets.QHBoxLayout(w1)
        l1.addWidget(self.table_faults)
        l1.addWidget(self.table_tests)
        splitter.addWidget(w1)

        splitter.addWidget(self.table_D)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(splitter)

        # signals
        self.btn_gen.clicked.connect(self.on_generate)
        self.btn_save.clicked.connect(self.on_save)
        self.btn_open.clicked.connect(self.on_open)

        # initial
        self.on_generate()

    def get_dataset_ref(self) -> Optional[Dataset]:
        return self.ds

    def on_generate(self):
        m = int(self.spin_m.value())
        n = int(self.spin_n.value())
        dens = float(self.spin_density.value())
        self.ds = random_dataset(m, n, density=dens, seed=None)
        self.populate_tables()

    def populate_tables(self):
        ds = self.ds
        if ds is None:
            return
        m, n = ds.D.shape
        # Faults
        self.table_faults.setRowCount(m)
        self.table_faults.setColumnCount(2)
        self.table_faults.setHorizontalHeaderLabels(["故障名称", "概率"])
        for i in range(m):
            self.table_faults.setItem(i, 0, QtWidgets.QTableWidgetItem(ds.fault_names[i]))
            self.table_faults.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{ds.fault_probs[i]:.6f}"))
        self.table_faults.resizeColumnsToContents()

        # Tests
        self.table_tests.setRowCount(n)
        self.table_tests.setColumnCount(2)
        self.table_tests.setHorizontalHeaderLabels(["测试名称", "代价"])
        for j in range(n):
            self.table_tests.setItem(j, 0, QtWidgets.QTableWidgetItem(ds.test_names[j]))
            self.table_tests.setItem(j, 1, QtWidgets.QTableWidgetItem(f"{ds.test_costs[j]:.6f}"))
        self.table_tests.resizeColumnsToContents()

        # D matrix
        self.table_D.setRowCount(m)
        self.table_D.setColumnCount(n)
        self.table_D.setHorizontalHeaderLabels(ds.test_names)
        self.table_D.setVerticalHeaderLabels(ds.fault_names)
        for i in range(m):
            for j in range(n):
                self.table_D.setItem(i, j, QtWidgets.QTableWidgetItem(str(int(ds.D[i, j]))))
        self.table_D.resizeColumnsToContents()

        # 可编辑
        self.table_faults.itemChanged.connect(self.on_fault_table_changed)
        self.table_tests.itemChanged.connect(self.on_test_table_changed)
        self.table_D.itemChanged.connect(self.on_D_changed)

    def on_fault_table_changed(self, item):
        if not self.ds:
            return
        row, col = item.row(), item.column()
        if col == 0:
            self.ds.fault_names[row] = item.text().strip()
            self.table_D.setVerticalHeaderItem(row, QtWidgets.QTableWidgetItem(self.ds.fault_names[row]))
        elif col == 1:
            try:
                val = float(item.text())
            except Exception:
                return
            self.ds.fault_probs[row] = max(1e-12, val)
            # 归一化
            s = self.ds.fault_probs.sum()
            self.ds.fault_probs = self.ds.fault_probs / s
            # 刷新显示
            self.table_faults.blockSignals(True)
            for i in range(self.ds.D.shape[0]):
                self.table_faults.setItem(i, 1, QtWidgets.QTableWidgetItem(f"{self.ds.fault_probs[i]:.6f}"))
            self.table_faults.blockSignals(False)

    def on_test_table_changed(self, item):
        if not self.ds:
            return
        row, col = item.row(), item.column()
        if col == 0:
            self.ds.test_names[row] = item.text().strip()
            self.table_D.setHorizontalHeaderItem(row, QtWidgets.QTableWidgetItem(self.ds.test_names[row]))
        elif col == 1:
            try:
                val = float(item.text())
            except Exception:
                return
            self.ds.test_costs[row] = max(1e-9, val)

    def on_D_changed(self, item):
        if not self.ds:
            return
        i, j = item.row(), item.column()
        try:
            v = int(item.text())
        except Exception:
            return
        v = 1 if v != 0 else 0
        self.ds.D[i, j] = v
        # 确保每个故障至少被覆盖一次（仅提示，不强制）
        if self.ds.D[i].sum() == 0:
            QtWidgets.QToolTip.showText(self.mapToGlobal(self.pos()), f"警告：{self.ds.fault_names[i]} 无任何覆盖测试。")

    def on_save(self):
        if not self.ds:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "保存 JSON", os.path.join(os.getcwd(), "data", "dataset.json"), "JSON (*.json)")
        if path:
            self.ds.save_json(path)
            QtWidgets.QMessageBox.information(self, "保存", f"已保存：{path}")

    def on_open(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "打开 JSON", os.path.join(os.getcwd(), "data"), "JSON (*.json)")
        if path:
            try:
                self.ds = Dataset.load_json(path)
                self.populate_tables()
                QtWidgets.QMessageBox.information(self, "打开", f"已加载：{path}")
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "错误", f"加载失败：{e}")
                