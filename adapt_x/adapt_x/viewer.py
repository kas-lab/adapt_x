#!/usr/bin/env python3

import sys

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTableWidget,
    QTableWidgetItem,
)

import numpy as np
import rclpy
from rclpy.node import Node

from adapt_x_msgs.msg import QTable


class QTableViewer(Node):

    def __init__(self, table):
        super().__init__("qtable_viewer")

        self.table = table
        self.initialized = False

        self.create_subscription(
            QTable,
            "/q_table",
            self.callback,
            10
        )

    def callback(self, msg):

        rows = msg.rows
        cols = msg.cols

        q = np.array(msg.values).reshape(rows, cols)

        if not self.initialized:

            self.table.setRowCount(rows)
            self.table.setColumnCount(cols)

            header = self.table.verticalHeader()

            for i, label in enumerate(msg.state_labels):
                item = QTableWidgetItem(f"S{i}")
                item.setToolTip(label)
                self.table.setVerticalHeaderItem(i, item)

            self.table.setHorizontalHeaderLabels(msg.action_labels)

            for r in range(rows):
                for c in range(cols):
                    item = QTableWidgetItem()
                    item.setTextAlignment(Qt.AlignCenter)
                    self.table.setItem(r, c, item)

            self.initialized = True

        self.update_table(q)

    def update_table(self, q):

        qmin = np.min(q)
        qmax = np.max(q)

        span = max(qmax - qmin, 1e-6)

        for r in range(q.shape[0]):
            for c in range(q.shape[1]):

                value = q[r, c]

                item = self.table.item(r, c)

                item.setText(f"{value:.3f}")

                t = (value - qmin) / span

                red = int((1.0 - t) * 255)
                green = int(t * 255)

                item.setBackground(QColor(red, green, 0))


def main():

    app = QApplication(sys.argv)

    table = QTableWidget()

    window = QMainWindow()
    window.setCentralWidget(table)
    window.resize(900, 700)
    window.setWindowTitle("Q Table Viewer")
    window.show()

    rclpy.init()

    node = QTableViewer(table)

    timer = QTimer()
    timer.timeout.connect(
        lambda: rclpy.spin_once(node, timeout_sec=0)
    )
    timer.start(30)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
