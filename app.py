# -*- coding: utf-8 -*-
import os
import sys
from PyQt5 import QtWidgets, uic

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from widgets.dataset_widget import DatasetWidget
from widgets.algos_widget import AlgosWidget
from widgets.benchmark_widget import BenchmarkWidget

def ensure_runtime_dirs():
    for d in ("data", "results"):
        p = os.path.join(BASE_DIR, d)
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)

def main():
    ensure_runtime_dirs()
    app = QtWidgets.QApplication(sys.argv)
    ui_path = os.path.join(BASE_DIR, "ui", "main_window.ui")
    MainWindow = uic.loadUi(ui_path)

    # 直接拿到已有的 tab 和它们的布局
    tab_dataset = MainWindow.findChild(QtWidgets.QWidget, "tabDataset")
    tab_algos   = MainWindow.findChild(QtWidgets.QWidget, "tabAlgorithms")
    tab_bench   = MainWindow.findChild(QtWidgets.QWidget, "tabBenchmark")

    layout_dataset = tab_dataset.layout()  # 对应 ui 中的 layoutDataset
    layout_algos   = tab_algos.layout()    # 对应 ui 中的 layoutAlgos
    layout_bench   = tab_bench.layout()    # 对应 ui 中的 layoutBench

    # 往已有布局里添加自定义组件（不要重新创建布局）
    dataset_widget = DatasetWidget(parent=tab_dataset)
    layout_dataset.setContentsMargins(6, 6, 6, 6)
    layout_dataset.addWidget(dataset_widget)

    algos_widget = AlgosWidget(parent=tab_algos, dataset_provider=dataset_widget.get_dataset_ref)
    layout_algos.setContentsMargins(6, 6, 6, 6)
    layout_algos.addWidget(algos_widget)

    bench_widget = BenchmarkWidget(parent=tab_bench, dataset_provider=dataset_widget.get_dataset_ref)
    layout_bench.setContentsMargins(6, 6, 6, 6)
    layout_bench.addWidget(bench_widget)

    MainWindow.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
