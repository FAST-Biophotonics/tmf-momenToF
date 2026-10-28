"""
tmf momenTOF - real-time acquisition, calibration, and batch moments fitting for the TMF882x.

    python tmf_momenTOF.py

Tabs
    Live Acquisition   connect to the EVM GUI (or replay a recording), acquire, record, calibrate, and fit μa / μs′ in real time
    Data Viewer        inspect recordings, replay frames, and explore fitted optical properties
    Offline Analysis   run batch moments analysis on sets of recordings, with summary tables and publication figures
"""
import os
import sys

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt

HERE = os.path.dirname(os.path.abspath(__file__))
CORE = os.path.join(HERE, "core")
for p in (HERE, CORE):
    if p not in sys.path:
        sys.path.insert(0, p)

from gui.styles import (
    C, QSS, APP_NAME, _PG_SYM, pill, set_pill, card, subcard, muted,
    rgba, style_plot, icon_button, CHANNEL_COLORS
)
from gui.persistence import (
    Persist, widget_values, set_widget_values, read_app_json, write_settings
)
from gui.widgets import (
    _ClickEdit, PathEdit, Collapsible, UnitForm, form_layout, dspin, ispin, unit_box,
    NoWheelOnNumbers, HistControls, LimitsDialog, PlotTools, plot_bar, with_bar,
    RefPair, CalForm, TabStack, read_csv, fill_table, new_table, new_log,
    scroll, hist_x, hist_y, is_recording, recording_channel_data,
    recording_hist, read_values_file, base, cal_summary
)
from gui.offline_tab import OfflineTab, Runner
from gui.live_tab import LiveTab, LiveOpticalWindow, ChannelPicker, Kpi
from gui.viewer_tab import DataViewerTab

# ----------------------------------------------------------------------------- main window
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}")
        self.resize(1500, 930)
        self.persist = Persist()
        root = QtWidgets.QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        v = QtWidgets.QVBoxLayout(root)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)

        head = QtWidgets.QWidget()
        head.setStyleSheet(f"background: {C['panel']}; border-bottom: 1px solid {C['border']};")
        hh = QtWidgets.QHBoxLayout(head)
        hh.setContentsMargins(20, 10, 20, 0)
        logo = QtWidgets.QLabel("◉")
        logo.setStyleSheet(f"color: {C['accent']}; font-size: 17pt; border: none;")
        title = QtWidgets.QLabel(APP_NAME)
        title.setObjectName("title")
        title.setStyleSheet("border: none;")
        sub = QtWidgets.QLabel("TMF882x-Compatible Software for Time-Domain Diffuse Optics")
        sub.setObjectName("muted")
        sub.setStyleSheet("border: none;")
        tbox = QtWidgets.QVBoxLayout()
        tbox.setSpacing(1)
        tbox.addWidget(title)
        tbox.addWidget(sub)
        hh.addWidget(logo, 0, Qt.AlignVCenter)
        hh.addSpacing(8)
        hh.addLayout(tbox)
        hh.addSpacing(36)

        self.bar = QtWidgets.QTabBar()
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        self.bar.setStyleSheet("border: none;")
        hh.addWidget(self.bar, 0, Qt.AlignBottom)
        hh.addStretch(1)

        self.state = pill("Disconnected", C["bad"])
        self.state.setObjectName("connPill")
        set_pill(self.state, "Disconnected", C["bad"])
        hh.addWidget(self.state, 0, Qt.AlignVCenter)
        self.state.setToolTip("Sensor connection status (Live Acquisition)")
        v.addWidget(head)

        self.live = LiveTab(self.persist)
        self.data_viewer = DataViewerTab(self.persist, self)
        self.off = OfflineTab(self.persist)

        self.stack = QtWidgets.QStackedWidget()
        for wdg, name in (
            (self.live, "Live Acquisition"),
            (self.data_viewer, "Data Viewer"),
            (self.off, "Offline Analysis"),
        ):
            self.bar.addTab(name)
            self.stack.addWidget(wdg)
        self.bar.currentChanged.connect(self.stack.setCurrentIndex)
        v.addWidget(self.stack, 1)

        # Wire cross-tab workflows
        self.live.status.connect(lambda t, c: set_pill(self.state, t, c))
        self.off.request_cal_tab.connect(self.switch_to_cal)
        self.live.calibrated.connect(self._on_calibrated)

        g = self.persist.s.value("window/geometry")
        if g is not None:
            self.restoreGeometry(g)

    def switch_to_cal(self):
        self.bar.setCurrentIndex(0)
        self.live.open_calibration_form()

    def _on_calibrated(self, path):
        self.off.w["cal"].setText(path)

    def closeEvent(self, e):
        self.live.shutdown()
        self.data_viewer.shutdown()
        self.off.save_state()
        self.off._drop_temp()
        self.persist.s.setValue("window/geometry", self.saveGeometry())
        super().closeEvent(e)


def main():
    QtWidgets.QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    app._no_wheel = NoWheelOnNumbers()
    app.installEventFilter(app._no_wheel)
    pal = app.palette()
    for role, col in ((QtGui.QPalette.Window, C["bg"]), (QtGui.QPalette.Base, C["panel"]),
                      (QtGui.QPalette.Text, C["text"]), (QtGui.QPalette.WindowText, C["text"]),
                      (QtGui.QPalette.Button, C["card2"]), (QtGui.QPalette.ButtonText, C["text"]),
                      (QtGui.QPalette.Highlight, C["accent"]), (QtGui.QPalette.ToolTipBase, C["card2"])):
        pal.setColor(role, QtGui.QColor(col))
    app.setPalette(pal)
    app.setStyleSheet(QSS)
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

