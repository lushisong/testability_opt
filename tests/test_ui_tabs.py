# -*- coding: utf-8 -*-
from pathlib import Path
import sys
import os
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PyQt5 import QtWidgets


def test_offline_tab_present(qtbot):
    # 确保 QApplication 存在
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    import app as appmod
    win = appmod.build_mainwindow()
    qtbot.addWidget(win)
    tab = win.findChild(QtWidgets.QTabWidget, "tabWidget")
    assert tab is not None
    titles = [tab.tabText(i) for i in range(tab.count())]
    assert any("离线训练" in t for t in titles)


def test_algos_widget_has_model_path(qtbot):
    from widgets.algos_widget import AlgosWidget
    w = AlgosWidget()
    qtbot.addWidget(w)
    # 切到离线算法，检查路径控件启用
    idx = w.cmb_algo.findText("NN-Guided_Offline")
    assert idx >= 0
    w.cmb_algo.setCurrentIndex(idx)
    assert w.txt_model_path.isEnabled()
    assert w.btn_model_browse.isEnabled()

