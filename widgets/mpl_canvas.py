# -*- coding: utf-8 -*-
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        self.setParent(parent)

class SimplePlot(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self.canvas)

    def plot_bar(self, labels, values, title, ylabel):
        self.canvas.ax.clear()
        x = range(len(labels))
        self.canvas.ax.bar(x, values)
        self.canvas.ax.set_xticks(list(x))
        self.canvas.ax.set_xticklabels(labels, rotation=0)
        self.canvas.ax.set_title(title)
        self.canvas.ax.set_ylabel(ylabel)
        self.canvas.draw()

    def show_image_file(self, path):
        import matplotlib.image as mpimg
        self.canvas.ax.clear()
        img = mpimg.imread(path)
        self.canvas.ax.imshow(img)
        self.canvas.ax.axis('off')
        self.canvas.draw()
