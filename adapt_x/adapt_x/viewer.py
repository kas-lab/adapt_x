#!/usr/bin/env python3

import argparse
import sys

import numpy as np
import rclpy
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
)
from rclpy.node import Node

from adapt_x_msgs.msg import QTable


class QTableViewer(Node):

    def __init__(self, table, npy_path=None):
        super().__init__("qtable_viewer")

        self.table = table
        self.initialized = False

        self.create_subscription(
            QTable,
            "/q_table",
            self.callback,
            10,
        )

        if npy_path is not None:
            self.load_from_npy(npy_path)

    def _default_labels(self, rows, cols):
        state_labels = [f"S{i}" for i in range(rows)]
        action_labels = [f"A{j}" for j in range(cols)]
        return state_labels, action_labels

    def initialize_table(self, rows, cols, state_labels=None, action_labels=None):
        if self.initialized:
            return

        self.table.setRowCount(rows)
        self.table.setColumnCount(cols)

        if state_labels is None:
            state_labels, _ = self._default_labels(rows, cols)
        if action_labels is None:
            _, action_labels = self._default_labels(rows, cols)

        for i, label in enumerate(state_labels):
            item = QTableWidgetItem(f"S{i}")
            item.setToolTip(label)
            self.table.setVerticalHeaderItem(i, item)

        self.table.setHorizontalHeaderLabels(action_labels)

        for r in range(rows):
            for c in range(cols):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                self.table.setItem(r, c, item)

        self.initialized = True

    def display_q_table(self, q, state_labels=None, action_labels=None):
        q = np.asarray(q, dtype=float)
        rows, cols = q.shape

        self.initialize_table(rows, cols, state_labels=state_labels, action_labels=action_labels)
        self.update_table(q)

    def callback(self, msg):
        rows = msg.rows
        cols = msg.cols

        q = np.array(msg.values).reshape(rows, cols)

        state_labels = list(msg.state_labels) if msg.state_labels else None
        action_labels = list(msg.action_labels) if msg.action_labels else None

        self.display_q_table(q, state_labels=state_labels, action_labels=action_labels)

    def load_from_npy(self, path):
        try:
            payload = np.load(path, allow_pickle=True)
        except Exception as exc:
            QMessageBox.critical(
                None,
                "Q-table load failed",
                f"Could not load {path}: {exc}",
            )
            return

        try:
            q = self._extract_q_table(payload)
        except ValueError as exc:
            QMessageBox.critical(
                None,
                "Q-table load failed",
                str(exc),
            )
            return

        if q.ndim != 2:
            QMessageBox.critical(
                None,
                "Q-table load failed",
                "The .npy file must contain a 2D array.",
            )
            return

        state_labels, action_labels = self._default_labels(q.shape[0], q.shape[1])
        self.display_q_table(q, state_labels=state_labels, action_labels=action_labels)

    def _extract_q_table(self, payload):
        if isinstance(payload, np.ndarray):
            return payload.astype(float, copy=False)

        if isinstance(payload, dict):
            for key in ("q_table", "values", "array", "data"):
                if key in payload:
                    return np.asarray(payload[key], dtype=float)

        if isinstance(payload, (tuple, list)) and payload:
            first = payload[0]
            if isinstance(first, np.ndarray):
                return np.asarray(first, dtype=float)
            if isinstance(first, (list, tuple)):
                return np.asarray(first, dtype=float)

        raise ValueError("Unsupported .npy format. Expected a 2D array or a simple dict/list payload.")

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
    parser = argparse.ArgumentParser(description="Visualize a Q-table from ROS or from a .npy file")
    parser.add_argument("--npy", type=str, help="Path to a .npy file containing a 2D Q-table")
    args = parser.parse_args()

    app = QApplication(sys.argv)

    table = QTableWidget()

    window = QMainWindow()
    window.setCentralWidget(table)
    window.resize(900, 700)
    window.setWindowTitle("Q Table Viewer")
    window.show()

    rclpy.init()

    node = QTableViewer(table, npy_path=args.npy)

    timer = QTimer()
    timer.timeout.connect(lambda: rclpy.spin_once(node, timeout_sec=0))
    timer.start(30)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
