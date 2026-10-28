"""Reusable UI controls, dialogs, form components, and plot tools."""
import csv
import datetime as dt
import json
import os
import re
import shlex
import shutil
import sys

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import pyqtgraph as pg

import figures as FIG
import tmf_moments_solver as V
from gui.styles import (
    C, APP_NAME, card, subcard, muted, pill, set_pill,
    style_plot, icon_button, _draw_limits, _draw_camera, _draw_html
)
from gui.persistence import widget_values, set_widget_values

try:
    import plotly
    _HAVE_PLOTLY = True
except ImportError:
    _HAVE_PLOTLY = False

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = ROOT_DIR

class _ClickEdit(QtWidgets.QLineEdit):
    clicked = QtCore.Signal()

    def mousePressEvent(self, e):
        self.clicked.emit()


class PathEdit(QtWidgets.QWidget):
    """A path chosen with the file / folder dialog only (no typing): click the field or Browse.
    kind: 'dir', 'file', 'save'. The full path is shown as a tooltip."""
    changed = QtCore.Signal(str)

    def __init__(self, kind="file", filt="CSV files (*.csv);;All files (*)", placeholder=""):
        super().__init__()
        self.kind, self.filt = kind, filt
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        self.edit = _ClickEdit()
        self.edit.setReadOnly(True)
        self.edit.setCursor(Qt.PointingHandCursor)
        self.edit.setPlaceholderText(placeholder or ("Select folder…" if kind == "dir" else "Select file…"))
        self.edit.clicked.connect(self.browse)
        b = QtWidgets.QPushButton("Browse…")
        b.setObjectName("small")
        b.clicked.connect(self.browse)
        self.clear_btn = QtWidgets.QPushButton("✕")
        self.clear_btn.setObjectName("small")
        self.clear_btn.setFixedWidth(28)
        self.clear_btn.setToolTip("Clear selection")
        self.clear_btn.clicked.connect(lambda: self.setText(""))
        self.edit.textChanged.connect(self._changed)
        h.addWidget(self.edit, 1)
        h.addWidget(b)
        h.addWidget(self.clear_btn)
        self.clear_btn.setVisible(False)

    def _changed(self, t):
        self.edit.setToolTip(t)
        self.clear_btn.setVisible(bool(t))
        self.edit.setCursorPosition(0)
        self.changed.emit(t)

    def browse(self):
        start = self.text() or os.path.expanduser("~")
        if self.kind == "dir":
            p = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", start)
        elif self.kind == "save":
            p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save As", start, self.filt)
        else:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select File", os.path.dirname(start) or start, self.filt)
        if p:
            self.edit.setText(os.path.normpath(p))

    def text(self):
        return self.edit.text().strip()

    def setText(self, t):
        self.edit.setText(t)


class Collapsible(QtWidgets.QWidget):
    def __init__(self, title, expanded=False):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 2, 0, 0)
        v.setSpacing(6)
        self.btn = QtWidgets.QToolButton()
        self.btn.setObjectName("section")
        self.btn.setCursor(Qt.PointingHandCursor)
        self.btn.setText(("▾  " if expanded else "▸  ") + title)
        self.btn.setCheckable(True)
        self.btn.setChecked(expanded)
        self.title = title
        self.body = QtWidgets.QWidget()
        self.form = UnitForm(self.body)
        self.form.setContentsMargins(6, 4, 0, 2)
        self.form.setSpacing(7)
        self.form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.body.setVisible(expanded)
        self.btn.toggled.connect(self._toggle)
        v.addWidget(self.btn)
        v.addWidget(self.body)

    def _toggle(self, on):
        self.body.setVisible(on)
        self.btn.setText(("▾  " if on else "▸  ") + self.title)

    def set_title(self, title):
        self.title = title
        self._toggle(self.btn.isChecked())


def unit_box(spin, unit=None):
    """A spin box with its unit as a label beside it (not inside the box)."""
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(0, 0, 0, 0)
    h.setSpacing(8)
    u = unit if unit is not None else spin.property("unit")
    lab = QtWidgets.QLabel(u or "")
    lab.setObjectName("muted")
    h.addWidget(spin, 1)
    h.addWidget(lab)
    spin.unit_container = w
    return w


class UnitForm(QtWidgets.QFormLayout):
    def addRow(self, *a):
        a = list(a)
        if len(a) == 2 and isinstance(a[1], QtWidgets.QAbstractSpinBox) and a[1].property("unit"):
            a[1] = unit_box(a[1])
        return super().addRow(*a)


def form_layout():
    f = UnitForm()
    f.setSpacing(7)
    f.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
    f.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
    return f


def dspin(lo, hi, val, dec=2, step=0.1, suffix=""):
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setProperty("unit", suffix)          # shown as a label beside the box (see UnitForm)
    return s


def ispin(lo, hi, val, suffix=""):
    s = QtWidgets.QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    if suffix:
        s.setProperty("unit", suffix)          # shown as a label beside the box (see UnitForm)
    return s




class Runner(QtCore.QObject):
    finished = QtCore.Signal(int)

    def __init__(self, log):
        super().__init__()
        self.log = log
        self.p = None

    def start(self, args):
        self.p = QtCore.QProcess(self)
        self.p.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        self.p.setWorkingDirectory(HERE)
        env = QtCore.QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("MPLBACKEND", "Agg")
        self.p.setProcessEnvironment(env)
        self.p.readyReadStandardOutput.connect(self._out)

        def _fin(code, _s):
            try:
                self.finished.emit(code)
            except (RuntimeError, AttributeError):
                pass

        def _err(e):
            try:
                if self.log:
                    self.log.appendPlainText(f"[could not start: {e}]")
            except (RuntimeError, AttributeError):
                pass

        self.p.finished.connect(_fin)
        self.p.errorOccurred.connect(_err)
        self.p.start(sys.executable, ["-u"] + args)

    def _out(self):
        try:
            if not self.log or not self.p:
                return
            t = bytes(self.p.readAllStandardOutput()).decode("utf-8", errors="replace")
            self.log.moveCursor(QtGui.QTextCursor.End)
            self.log.insertPlainText(t)
            self.log.moveCursor(QtGui.QTextCursor.End)
        except (RuntimeError, AttributeError):
            pass

    def running(self):
        try:
            return self.p is not None and self.p.state() != QtCore.QProcess.NotRunning
        except (RuntimeError, AttributeError):
            return False

    def kill(self):
        if self.running():
            try:
                self.p.readyReadStandardOutput.disconnect()
                self.p.finished.disconnect()
                self.p.errorOccurred.disconnect()
            except Exception:
                pass
            try:
                self.p.kill()
                self.p.waitForFinished(1000)
            except Exception:
                pass




def read_csv(path):
    with open(path, newline="", encoding="utf-8", errors="ignore") as fh:
        rows = list(csv.reader(fh))
    return (rows[0], rows[1:]) if rows else ([], [])


def fill_table(t, head, rows):
    t.clear()
    t.setColumnCount(len(head))
    t.setRowCount(len(rows))
    t.setHorizontalHeaderLabels(head)
    for i, r in enumerate(rows):
        for j, x in enumerate(r):
            it = QtWidgets.QTableWidgetItem(x)
            it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if j else Qt.AlignLeft | Qt.AlignVCenter)
            t.setItem(i, j, it)
    t.verticalHeader().setVisible(False)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)


def new_table():
    t = QtWidgets.QTableWidget()
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setAlternatingRowColors(True)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
    return t


def new_log():
    l = QtWidgets.QPlainTextEdit()
    l.setObjectName("log")
    l.setReadOnly(True)
    l.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
    return l


def scroll(widget, width=None):
    s = QtWidgets.QScrollArea()
    s.setWidgetResizable(True)
    s.setWidget(widget)
    s.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    if width:
        s.setFixedWidth(width)
        widget.setMaximumWidth(width - 14)
    widget.setObjectName("root")
    return s





# ----------------------------------------------------------------------------- channel picker
class ChannelPicker(QtWidgets.QWidget):
    """Grid of channel (zone) buttons, laid out like the sensor's zone map. Display only.

    Zone numbers follow the TMF882x datasheet (DS000693, Fig. 30/31): zone 1 top-left, row by row.
    3x3 = 3 columns x 3 rows, 4x4 = 4 x 4, 3x6 = 3 columns x 6 rows. #HLONGnn in the data is zone nn.
    The grid keeps its layout and your selection when data arrives; zones with no data are greyed out.
    """
    changed = QtCore.Signal()
    layout_from_data = QtCore.Signal(str, str)         # (layout name, reason) when the data needs another layout
    LAYOUTS = {"3 x 3": (3, 3), "4 x 4": (4, 4), "3 x 6": (6, 3), "8 x 8": (8, 8)}   # name -> (rows, cols)
    BY_COUNT = {9: "3 x 3", 16: "4 x 4", 18: "3 x 6", 64: "8 x 8"}

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        self.grid_box = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        v.addWidget(self.grid_box)
        h = QtWidgets.QHBoxLayout()
        b_all = QtWidgets.QPushButton("All")
        b_all.setObjectName("small")
        b_all.clicked.connect(lambda: self.set_all(True))
        b_none = QtWidgets.QPushButton("None")
        b_none.setObjectName("small")
        b_none.clicked.connect(lambda: self.set_all(False))
        h.addWidget(b_all)
        h.addWidget(b_none)
        h.addStretch(1)
        v.addLayout(h)
        self.overlay = QtWidgets.QCheckBox("Overlay individual channels")
        self.overlay.setChecked(False)
        self.overlay.toggled.connect(self.changed)
        v.addWidget(self.overlay)
        self.note = QtWidgets.QLabel("")
        self.note.setObjectName("muted")
        self.note.setWordWrap(True)
        self.note.setVisible(False)
        v.addWidget(self.note)
        self.buttons = {}
        self.layout_name = None
        self.data_chans = None          # set of zone numbers present in the data (None = no data yet)
        self.fit_zones = None           # zones the live fit uses (from the calibration), None = all

    # -- layout
    def set_layout(self, name):
        """Rebuild the grid for a pixel mode ('3 x 3', '4 x 4', '3 x 6', '8 x 8'), keeping the current selection."""
        if name not in self.LAYOUTS or name == self.layout_name:
            return
        rows, cols = self.LAYOUTS[name]
        self._build(list(range(1, rows * cols + 1)), cols)
        self.layout_name = name
        self._apply_data()
        self.changed.emit()

    def _build(self, zones, cols):
        old = {c: b.isChecked() for c, b in self.buttons.items()}
        all_on = not old or all(old.values())
        self._memory = {**getattr(self, "_memory", {}), **old}     # remember zones that leave the grid (3x3 <-> 4x4)
        old = {**self._memory, **old}
        for b in self.buttons.values():
            self.grid.removeWidget(b)
            b.deleteLater()
        self.buttons = {}
        for i, c in enumerate(zones):
            b = QtWidgets.QPushButton(str(c))
            b.setCheckable(True)
            b.setChecked(old.get(c, all_on))
            b.setObjectName("chan")
            b.setFixedHeight(26)
            b.toggled.connect(self.changed)
            self.grid.addWidget(b, i // cols, i % cols)
            self.buttons[c] = b

    def ensure(self, chans):
        """Called with the zone numbers of every incoming frame. Keeps the layout (and selection) when the data fits it;
        otherwise switches to the layout that matches the data and says why."""
        chans = {int(c) for c in chans}
        if chans == self.data_chans and chans <= set(self.buttons) and not (
                self.layout_name == "8 x 8" and chans and max(chans) <= 18):
            return
        self.data_chans = chans
        if chans:
            zones = set(self.buttons)
            exact = self.BY_COUNT.get(len(chans)) if chans == set(range(1, len(chans) + 1)) else None
            if (exact and exact != self.layout_name) or not chans <= zones or (
                    self.layout_name == "8 x 8" and max(chans) <= 18):
                name = exact or self.BY_COUNT.get(len(chans)) or next(
                    (n for n in ("3 x 3", "4 x 4", "3 x 6", "8 x 8")
                     if max(chans) <= self.LAYOUTS[n][0] * self.LAYOUTS[n][1] and min(chans) >= 1), None)
                if name:
                    r_, c_ = self.LAYOUTS[name]
                    if not chans <= set(range(1, r_ * c_ + 1)):
                        name = None                 # e.g. zero-based channel names: no datasheet layout fits
                if name and name != self.layout_name:
                    was = self.layout_name
                    self.set_layout(name)
                    why = (f"The data has {len(chans)} channels, so the grid shows {name.replace(' x ', '×')}"
                           + (f" (was {was.replace(' x ', '×')})." if was else "."))
                    if was == "8 x 8":
                        why += " 8×8 data arrives as 16-zone sub-captures, so each channel here is one sub-capture zone."
                    self.layout_from_data.emit(name, why)
                elif not name:                      # unknown channel set: plain grid of what is there
                    self._build(sorted(chans), 4)
                    self.layout_name = None
        self._apply_data()
        self.changed.emit()

    def _apply_data(self):
        for c, b in self.buttons.items():
            has = self.data_chans is None or c in self.data_chans
            b.setEnabled(has)
            tip = f"Zone {c}"
            if not has:
                tip += " · no data for this zone in the current stream"
            if self.fit_zones is not None and c in self.fit_zones:
                tip += " · used by the live fit (calibration zones)"
            b.setToolTip(tip)
            b.setProperty("fit", bool(self.fit_zones is not None and c in self.fit_zones))
            b.style().unpolish(b)
            b.style().polish(b)

    def fits(self, name, chans=None):
        """True if layout `name` can show these channels (default: the channels in the current data)."""
        chans = self.data_chans if chans is None else chans
        if not chans or name not in self.LAYOUTS:
            return True
        r, c = self.LAYOUTS[name]
        if not chans <= set(range(1, r * c + 1)):
            return False
        if name == "8 x 8" and max(chans) <= 18:
            return False
        exact = self.BY_COUNT.get(len(chans)) if chans == set(range(1, len(chans) + 1)) else None
        return exact in (None, name)

    def set_fit_zones(self, zones):
        """Zones the live fit uses (calibration's --zones), or None for all. Shown as a note and on the buttons."""
        self.fit_zones = {int(z) for z in zones} if zones else None
        if self.fit_zones:
            self.note.setText("Live fit uses zones " + ", ".join(map(str, sorted(self.fit_zones)))
                              + " (from the calibration, outlined). The selection above only changes the plot.")
        else:
            self.note.setText("")
        self.note.setVisible(bool(self.fit_zones))
        self._apply_data()

    def reset_data(self):
        """New stream: forget which zones had data (the layout and selection are kept)."""
        self.data_chans = None
        self._apply_data()

    def set_all(self, on):
        for b in self.buttons.values():
            b.blockSignals(True)
            b.setChecked(on)
            b.blockSignals(False)
        self.changed.emit()

    def selected(self):
        return [c for c, b in sorted(self.buttons.items()) if b.isChecked() and b.isEnabled()]

    def describe(self):
        sel = self.selected()
        avail = [c for c, b in self.buttons.items() if b.isEnabled()]
        lay = f"{self.layout_name.replace(' x ', '×')} · " if self.layout_name else ""
        if not self.buttons:
            return "all channels"
        if len(sel) == len(avail):
            what = f"all {len(sel)} channels"
        elif not sel:
            return lay + "no channels selected"
        elif len(sel) == 1:
            what = f"channel {sel[0]}"
        else:
            what = "channels " + ", ".join(map(str, sel))
        return lay + what + (" (overlay)" if self.overlay.isChecked() and len(sel) > 1 else " (summed)" if len(sel) > 1 else "")


# ----------------------------------------------------------------------------- plot tools
def hist_x(n, unit):
    """x values of n histogram bins: bin index, or bin centre in ns."""
    return np.arange(n, dtype=float) if unit == "bins" else (np.arange(n) + 0.5) * V.DT_PS / 1000.0


def hist_y(y, norm, log):
    """Normalise ('none' | 'peak' | 'area') and keep values positive for a log axis."""
    y = np.asarray(y, dtype=float)
    if norm == "peak":
        m = np.nanmax(y)
        y = y / m if m > 0 else y
    elif norm == "area":
        a = np.nansum(np.clip(y, 0, None))
        y = y / a if a > 0 else y
    if log:
        top = np.nanmax(y) if np.isfinite(y).any() else 1.0
        y = np.clip(y, 1.0 if norm == "none" else max(top, 1e-12) * 1e-5, None)
    return y


class HistControls(QtWidgets.QWidget):
    """Log / normalise / x-axis unit / display options for a histogram plot (matching AMS EVM GUI)."""
    changed = QtCore.Signal()
    snapshot_clicked = QtCore.Signal()

    def __init__(self):
        super().__init__()
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)

        self.meas_cb = QtWidgets.QCheckBox("Meas")
        self.meas_cb.setChecked(True)
        self.meas_cb.setToolTip("Show/hide measurement signal histogram (c_h)")

        self.ref_cb = QtWidgets.QCheckBox("HOCAL")
        self.ref_cb.setChecked(False)
        self.ref_cb.setToolTip("Show/hide internal HOCAL reference pulse")

        self.log_cb = QtWidgets.QCheckBox("Log Y")
        self.log_cb.setChecked(True)

        self.autoscale_cb = QtWidgets.QCheckBox("Autoscale")
        self.autoscale_cb.setChecked(True)
        self.autoscale_cb.setToolTip("Automatically scale histogram Y-axis (uncheck to set fixed max)")

        self.ymax_spin = ispin(100, 10000000, 100000)
        self.ymax_spin.setFixedWidth(80)
        self.ymax_spin.setEnabled(False)
        self.ymax_spin.setToolTip("Fixed Y-axis maximum when Autoscale is unchecked (EVM GUI default 100000)")

        self.legend_cb = QtWidgets.QCheckBox("Legend")
        self.legend_cb.setChecked(True)

        self.norm_cb = QtWidgets.QComboBox()
        self.norm_cb.addItems(["No normalisation", "Peak-normalised", "Area-normalised"])
        self.norm_cb.setToolTip(
            "Histogram normalization mode:\n"
            "• No normalisation: Display absolute photon counts\n"
            "• Peak-normalised: Scaled so peak value = 1.0\n"
            "• Area-normalised: Scaled so total curve area = 1.0"
        )

        self.x_cb = QtWidgets.QComboBox()
        self.x_cb.addItems(["Time (ns)", "Time bin"])

        self.curve_cb = QtWidgets.QComboBox()
        self.curve_cb.addItems([
            "Background corrected (fitted)",
            "Raw counts",
        ])
        self.curve_cb.setToolTip(
            "Select histogram curve version:\n"
            "• Background corrected (fitted): Background model (DL crosstalk / pedestal) subtracted for moments fitting\n"
            "• Raw counts: Untouched detector histogram with dark/ambient floor"
        )

        for wdg in (self.meas_cb, self.ref_cb, self.log_cb, self.autoscale_cb, self.ymax_spin,
                    self.legend_cb, self.norm_cb, self.x_cb, self.curve_cb):
            h.addWidget(wdg)

        self.meas_cb.toggled.connect(self.changed)
        self.ref_cb.toggled.connect(self.changed)
        self.log_cb.toggled.connect(self.changed)
        self.autoscale_cb.toggled.connect(lambda on: (self.ymax_spin.setEnabled(not on), self.changed.emit()))
        self.ymax_spin.valueChanged.connect(self.changed)
        self.legend_cb.toggled.connect(self.changed)
        self.norm_cb.currentIndexChanged.connect(self.changed)
        self.x_cb.currentIndexChanged.connect(self.changed)
        self.curve_cb.currentIndexChanged.connect(self.changed)

    def log(self):
        return self.log_cb.isChecked()

    def norm(self):
        return ("none", "peak", "area")[self.norm_cb.currentIndex()]

    def unit(self):
        return "bins" if self.x_cb.currentIndex() == 1 else "ns"

    def curve_mode(self):
        modes = ("clean", "raw")
        idx = self.curve_cb.currentIndex()
        return modes[idx] if 0 <= idx < len(modes) else "clean"

    def set_curve_mode(self, mode):
        modes = {"clean": 0, "floor": 0, "bg_corrected": 0, "bg": 0, "overlay": 0, "raw": 1}
        if mode in modes:
            self.curve_cb.setCurrentIndex(modes[mode])

    def show_meas(self):
        return self.meas_cb.isChecked()

    def show_ref(self):
        return self.ref_cb.isChecked()

    def apply_axes(self, plot):
        plot.setLogMode(False, self.log())
        plot.setLabel("left", {"none": "photon counts", "peak": "normalised counts (peak = 1)",
                               "area": "normalised counts (area = 1)"}[self.norm()])
        plot.setLabel("bottom", "bin index" if self.unit() == "bins" else "time (ns)")
        left_axis = plot.getAxis("left")
        if hasattr(left_axis, "norm_mode"):
            left_axis.norm_mode = self.norm()
        if not self.autoscale_cb.isChecked():
            val = float(self.ymax_spin.value())
            if self.log():
                plot.setYRange(0, np.log10(max(val, 10.0)), padding=0)
            else:
                plot.setYRange(0, val, padding=0)
        else:
            plot.enableAutoRange(axis="y")
        if hasattr(plot, "legend") and plot.legend:
            plot.legend.setVisible(self.legend_cb.isChecked())


class LimitsDialog(QtWidgets.QDialog):
    """Type the axis limits of a plot instead of zooming with the mouse. Blank = automatic."""

    def __init__(self, parent, plots):
        super().__init__(parent)
        self.setWindowTitle("Axis Limits")
        self.plots = plots
        v = QtWidgets.QVBoxLayout(self)
        f = form_layout()
        self.which = QtWidgets.QComboBox()
        self.which.addItems([n for n, _ in plots])
        if len(plots) > 1:
            f.addRow("Panel", self.which)
        self.e = {k: QtWidgets.QLineEdit() for k in ("x0", "x1", "y0", "y1")}
        for k, e in self.e.items():
            e.setPlaceholderText("auto")
        hx = QtWidgets.QHBoxLayout()
        hx.addWidget(self.e["x0"])
        hx.addWidget(QtWidgets.QLabel("to"))
        hx.addWidget(self.e["x1"])
        hy = QtWidgets.QHBoxLayout()
        hy.addWidget(self.e["y0"])
        hy.addWidget(QtWidgets.QLabel("to"))
        hy.addWidget(self.e["y1"])
        f.addRow("X range", hx)
        f.addRow("Y range", hy)
        f.addRow(muted("Leave blank for automatic scaling."))
        v.addLayout(f)
        hb = QtWidgets.QHBoxLayout()
        b_auto = QtWidgets.QPushButton("Auto-scale")
        b_auto.setToolTip("Reset to automatic axis scaling")
        b_apply = QtWidgets.QPushButton("Apply")
        b_apply.setObjectName("primary")
        b_close = QtWidgets.QPushButton("Close")
        hb.addWidget(b_auto)
        hb.addStretch(1)
        hb.addWidget(b_close)
        hb.addWidget(b_apply)
        v.addLayout(hb)
        b_auto.clicked.connect(self.auto)
        b_apply.clicked.connect(self.apply)
        b_close.clicked.connect(self.close)
        self.which.currentIndexChanged.connect(self.fill)
        self.fill()

    def plot(self):
        return self.plots[self.which.currentIndex()][1]

    def _logy(self):
        return self.plot().getAxis("left").logMode

    def fill(self, *_):
        (x0, x1), (y0, y1) = self.plot().viewRange()
        if self._logy():
            y0, y1 = 10 ** y0, 10 ** y1
        for k, val in zip(("x0", "x1", "y0", "y1"), (x0, x1, y0, y1)):
            self.e[k].setText(f"{val:.4g}")

    def apply(self):
        p = self.plot()

        def num(k):
            t = self.e[k].text().strip()
            return float(t) if t else None

        try:
            x0, x1, y0, y1 = (num(k) for k in ("x0", "x1", "y0", "y1"))
        except ValueError:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Limits must be numeric values (or blank for auto-scaling).")
            return
        if x0 is not None and x1 is not None and x1 > x0:
            p.setXRange(x0, x1, padding=0)
        else:
            p.enableAutoRange(axis="x")
        if y0 is not None and y1 is not None and y1 > y0:
            if self._logy():
                if y0 <= 0:
                    QtWidgets.QMessageBox.warning(self, APP_NAME, "Log-axis limits must be greater than 0.")
                    return
                y0, y1 = np.log10(y0), np.log10(y1)
            p.setYRange(y0, y1, padding=0)
        else:
            p.enableAutoRange(axis="y")

    def auto(self):
        self.plot().enableAutoRange()
        QtCore.QTimer.singleShot(50, self.fill)




class NoWheelOnNumbers(QtCore.QObject):
    """Numbers change only by typing (or the arrows): the mouse wheel over a number box scrolls the page instead."""

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Wheel and isinstance(obj, QtWidgets.QAbstractSpinBox):
            w = obj.parentWidget()
            while w is not None and not isinstance(w, QtWidgets.QAbstractScrollArea):
                w = w.parentWidget()
            if w is not None:
                sb = w.verticalScrollBar()
                sb.setValue(sb.value() - ev.angleDelta().y() // 2)
            return True
        return False


def _plain(t):
    return re.sub(r"<[^>]+>", "", t or "").strip()


def _ink(qc):
    """curve colour for a white page: very light colours (meant for the dark app) become dark ink"""
    return FIG.INK if qc.lightness() > 200 else qc.name()


def plot_panels(plots):
    """what a set of pyqtgraph plots is showing, as data for figures.generic_png / generic_html"""
    panels = []
    for _name, p in plots:
        left, bottom = p.getAxis("left"), p.getAxis("bottom")
        logx, logy = bool(bottom.logMode), bool(left.logMode)
        (x0, x1), (y0, y1) = p.viewRange()
        if logx:
            x0, x1 = 10 ** x0, 10 ** x1
        if logy:
            y0, y1 = 10 ** y0, 10 ** y1
        pn = {"title": _plain(p.titleLabel.text) if p.titleLabel.isVisible() else "",
              "ylabel": _plain(left.labelText) + (f" ({left.labelUnits})" if left.labelUnits else ""),
              "xlabel": _plain(bottom.labelText) + (f" ({bottom.labelUnits})" if bottom.labelUnits else ""),
              "logx": logx, "logy": logy, "curves": [], "hlines": []}
        auto_x, auto_y = p.getViewBox().autoRangeEnabled()
        if not auto_x:                      # zoomed in or limits typed: keep that view
            pn["xrange"] = (x0, x1)
        if not auto_y:
            pn["yrange"] = (y0, y1)
        for it in p.listDataItems():
            x, y = it.xData, it.yData
            if x is None or y is None or len(x) == 0 or not np.isfinite(np.asarray(y, float)).any():
                continue
            pen = pg.mkPen(it.opts.get("pen")) if it.opts.get("pen") is not None else None
            line = pen is not None and pen.style() != Qt.NoPen and pen.color().alpha() > 0
            sym = it.opts.get("symbol")
            sb = it.opts.get("symbolBrush")
            col = _ink(pen.color()) if line else (_ink(pg.mkBrush(sb).color()) if sb is not None else FIG.INK)
            pn["curves"].append({"x": np.asarray(x, float), "y": np.asarray(y, float), "name": it.name() or "",
                                 "colour": col, "width": max(1.0, pen.widthF()) if line else 0,
                                 "dashed": bool(line and pen.style() == Qt.DashLine), "line": line,
                                 "marker": sym if isinstance(sym, str) else ("o" if sym is not None else None),
                                 "size": float(it.opts.get("symbolSize") or 5)})
        for it in p.items:
            if isinstance(it, pg.InfiniteLine) and it.angle == 0:
                pn["hlines"].append((float(it.value()), _ink(it.pen.color()), it.pen.style() == Qt.DashLine))
        if pn["curves"] or pn["hlines"]:
            panels.append(pn)
    return panels




class PlotTools(QtWidgets.QWidget):
    """Limits, PNG (camera) and interactive HTML (</>) for one plot area."""

    def __init__(self, area, plots, name="plot", png=None, html=None):
        super().__init__()
        self.area, self.plots, self.name = area, plots, name
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        b1 = icon_button(_draw_limits, "Set axis limits (or reset to auto-scale)")
        b1.clicked.connect(self.limits)
        b2 = icon_button(_draw_camera, "Export publication-ready PNG figure", "PNG")
        b2.clicked.connect(png or (lambda: self.save("png")))
        b3 = icon_button(_draw_html, "Export interactive HTML figure (Plotly)", "HTML")
        b3.clicked.connect(html or (lambda: self.save("html")))
        h.addWidget(b1)
        h.addWidget(b2)
        h.addWidget(b3)
        self.dlg = None

    def limits(self):
        if not self.plots:
            return
        self.dlg = LimitsDialog(self, self.plots)
        self.dlg.show()

    def save(self, fmt):
        if fmt == "html" and not _HAVE_PLOTLY:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "Interactive HTML export requires plotly. Install it once:\n\n"
                "    pip install plotly\n\nthen restart the application."
            )
            return
        panels = plot_panels(self.plots)
        if not panels:
            QtWidgets.QMessageBox.information(self, APP_NAME, "Nothing to export yet — the plot is empty.")
            return
        st = QtCore.QSettings("FAST-Biophotonics", "pToF-Moments")
        d = st.value("snap_dir", "") or os.path.expanduser("~")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Figure as " + ("PNG" if fmt == "png" else "Interactive HTML"),
            os.path.join(d, f"{self.name}_{dt.datetime.now():%Y%m%d_%H%M%S}.{fmt}"),
            "PNG image (*.png)" if fmt == "png" else "Interactive HTML (*.html)")
        if not p:
            return
        if not p.lower().endswith("." + fmt):
            p += "." + fmt
        try:
            (FIG.generic_png if fmt == "png" else FIG.generic_html)(panels, p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not export figure: {e}")
            return
        st.setValue("snap_dir", os.path.dirname(p))


def plot_bar(left_widgets, tools):
    """A row above a plot: controls on the left, Limits / Export on the right."""
    w = QtWidgets.QWidget()
    h = QtWidgets.QHBoxLayout(w)
    h.setContentsMargins(4, 0, 4, 0)
    h.setSpacing(8)
    for x in left_widgets:
        h.addWidget(x)
    h.addStretch(1)
    h.addWidget(tools)
    return w


def with_bar(bar, area):
    box = QtWidgets.QFrame()
    box.setObjectName("card")
    v = QtWidgets.QVBoxLayout(box)
    v.setContentsMargins(10, 8, 10, 10)
    v.setSpacing(6)
    v.addWidget(bar)
    v.addWidget(area, 1)
    return box



_HIST_CACHE = {}


def is_recording(path):
    """True for an EVM GUI log (#ITT / #HLONG rows), False for e.g. a CSV of certified values."""
    try:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for i, line in enumerate(fh):
                if line.startswith(("#ITT", "#HLONG", "#HOCAL", "#HECAL", "#VER", "#TMF")):
                    return True
                if i > 400:
                    break
    except OSError:
        return False
    return False


_CHAN_CACHE = {}


def recording_channel_data(path):
    """Return ({channel_int: 1d_hist_array}, hocal_ref_or_None) for a recording, averaged over its frames, floor removed."""
    if not path or not os.path.isfile(path):
        return {}, None
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return {}, None
    if key not in _CHAN_CACHE:
        z, _ = V.read_hlong(path)
        chans = {}
        if z:
            for k, v in z.items():
                if not k.startswith("#HLONG"):
                    continue
                digits = re.findall(r"\d+", k)
                ch = int(digits[0]) if digits else 0
                y = np.asarray(v, dtype=float)
                if len(y) >= 12:
                    y = y - np.median(y[:12])
                chans[ch] = y
        try:
            ref = V.read_hocal(path)
        except BaseException:
            ref = None
        _CHAN_CACHE[key] = (chans, ref)
    return _CHAN_CACHE[key]


def recording_hist(path):
    """(sum of all channels, HOCAL reference or None) of a recording, averaged over its frames, floor removed."""
    chans, ref = recording_channel_data(path)
    if not chans:
        return None, ref
    max_len = max(len(v) for v in chans.values())
    total = np.zeros(max_len, dtype=float)
    for arr in chans.values():
        total[:len(arr)] += arr
    return total, ref


def read_values_file(p):
    """certified values: {name: (mua, musp)} from a .json (milanfit_expected / certified_values) or a CSV"""
    if p.lower().endswith(".json"):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        return {k: (v["mean_mua"], v["mean_musp"]) for k, v in d.items() if isinstance(v, dict)}
    return {k: tuple(v[:2]) for k, v in V.read_compare(p).items()}


def _match(stem, vals):
    return stem if stem in vals else stem.split("_")[0] if stem.split("_")[0] in vals else None


class RefPair(QtWidgets.QWidget):
    """The calibration's reference phantom, and optionally a second one that is used instead whenever the first is
    itself in the data (so it is never fitted with its own calibration). Each: recording + certified mua, musp."""
    changed = QtCore.Signal()
    TITLES = ("Primary Reference Phantom", "Secondary Reference Phantom")

    def __init__(self):
        super().__init__()
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.slots = []
        val = QtGui.QDoubleValidator(0.0, 1000.0, 6)
        val.setNotation(QtGui.QDoubleValidator.StandardNotation)

        # Slot 0: Primary Reference
        t0 = QtWidgets.QLabel(self.TITLES[0])
        t0.setObjectName("sectionTitle")
        v.addWidget(t0)
        pe0 = PathEdit("file", "CSV files (*.csv);;All files (*)", "Select primary reference recording (.csv)…")
        v.addWidget(pe0)
        h0 = QtWidgets.QHBoxLayout()
        h0.setSpacing(6)
        mua0, musp0 = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e, ph in ((mua0, "certified μa"), (musp0, "certified μs′")):
            e.setValidator(val)
            e.setPlaceholderText(ph)
            e.textChanged.connect(lambda *_: self.changed.emit())
        for lab, e in (("μa", mua0), ("μs′", musp0)):
            l = QtWidgets.QLabel(lab)
            l.setObjectName("muted")
            h0.addWidget(l)
            h0.addWidget(e, 1)
        u0 = QtWidgets.QLabel("cm⁻¹")
        u0.setObjectName("muted")
        h0.addWidget(u0)
        v.addLayout(h0)
        pe0.changed.connect(lambda p: self._path_changed(0, p))
        self.slots.append((pe0, mua0, musp0))

        v.addSpacing(4)

        # Secondary Reference Phantom (optional checkbox)
        self.enable_sec = QtWidgets.QCheckBox("Include Secondary Reference Phantom")
        v.addWidget(self.enable_sec)

        self.sec_box = QtWidgets.QWidget()
        sec_lay = QtWidgets.QVBoxLayout(self.sec_box)
        sec_lay.setContentsMargins(0, 0, 0, 0)
        sec_lay.setSpacing(6)

        t1 = QtWidgets.QLabel(self.TITLES[1])
        t1.setObjectName("muted")
        sec_lay.addWidget(t1)
        pe1 = PathEdit("file", "CSV files (*.csv);;All files (*)", "Select secondary reference recording (.csv)…")
        sec_lay.addWidget(pe1)
        h1 = QtWidgets.QHBoxLayout()
        h1.setSpacing(6)
        mua1, musp1 = QtWidgets.QLineEdit(), QtWidgets.QLineEdit()
        for e, ph in ((mua1, "certified μa"), (musp1, "certified μs′")):
            e.setValidator(val)
            e.setPlaceholderText(ph)
            e.textChanged.connect(lambda *_: self.changed.emit())
        for lab, e in (("μa", mua1), ("μs′", musp1)):
            l = QtWidgets.QLabel(lab)
            l.setObjectName("muted")
            h1.addWidget(l)
            h1.addWidget(e, 1)
        u1 = QtWidgets.QLabel("cm⁻¹")
        u1.setObjectName("muted")
        h1.addWidget(u1)
        sec_lay.addLayout(h1)
        pe1.changed.connect(lambda p: self._path_changed(1, p))
        self.slots.append((pe1, mua1, musp1))

        self.sec_box.setVisible(False)
        self.enable_sec.toggled.connect(self.sec_box.setVisible)
        self.enable_sec.toggled.connect(lambda *_: self.changed.emit())
        v.addWidget(self.sec_box)

        # Load certified button
        h_btn = QtWidgets.QHBoxLayout()
        h_btn.addStretch(1)
        b_fill = QtWidgets.QPushButton("Load Certified Values (.json / .csv)…")
        b_fill.setObjectName("small")
        b_fill.setToolTip("Load certified optical properties from a JSON or CSV file (matched by phantom filename, e.g. B3.csv → B3).")
        b_fill.clicked.connect(lambda: self.fill_from_file())
        h_btn.addWidget(b_fill)
        v.addLayout(h_btn)

    def _path_changed(self, i, p):
        if p and os.path.isfile(p) and not is_recording(p):
            self.slots[i][0].blockSignals(True)
            self.slots[i][0].setText("")
            self.slots[i][0].blockSignals(False)
            if QtWidgets.QMessageBox.question(
                    self, APP_NAME,
                    f"{os.path.basename(p)} is not a sensor recording — it appears to be a table of "
                    "certified optical properties.\n\nUse it to populate μa and μs′?") == QtWidgets.QMessageBox.Yes:
                self.fill_from_file(p)
            return
        if p:                         # values kept next to recordings made in this app
            cv = os.path.join(os.path.dirname(p), "certified_values.json")
            if os.path.isfile(cv) and not self.slots[i][1].text() and not self.slots[i][2].text():
                try:
                    k = _match(os.path.splitext(os.path.basename(p))[0], read_values_file(cv))
                    if k:
                        vals = read_values_file(cv)[k]
                        self.slots[i][1].setText(f"{vals[0]:g}")
                        self.slots[i][2].setText(f"{vals[1]:g}")
                except Exception:
                    pass
        self.changed.emit()

    def fill_from_file(self, p=None):
        if not p:
            p, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, "Load Certified Phantom Values", "",
                "Certified values (*.json *.csv);;All files (*)"
            )
        if not p:
            return
        try:
            vals = read_values_file(p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not read {os.path.basename(p)}: {e}")
            return
        miss = []
        for pe, mua, musp in self.slots:
            if not pe.text():
                continue
            stem = os.path.splitext(os.path.basename(pe.text()))[0]
            k = _match(stem, vals)
            if k is None:
                miss.append(stem)
                continue
            mua.setText(f"{vals[k][0]:g}")
            musp.setText(f"{vals[k][1]:g}")
        if miss:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "No matching entry found for: " + ", ".join(miss) + "\nPlease enter μa and μs′ manually."
            )

    def slot(self, i):
        pe, mua, musp = self.slots[i]
        return [pe.text(), mua.text().strip(), musp.text().strip()]

    def rows(self):
        """[[path, mua, musp]] of the filled slots: first = the calibration's reference, second = the extra one"""
        r = [self.slot(0)] if self.slot(0)[0] else []
        if self.enable_sec.isChecked() and self.slot(1)[0]:
            r.append(self.slot(1))
        return r

    def set_rows(self, rows):
        rows = list(rows)[:2] + [["", "", ""]] * (2 - min(2, len(rows)))
        if rows[1][0]:
            self.enable_sec.setChecked(True)
        for (pe, mua, musp), (p, a, b) in zip(self.slots, rows):
            pe.blockSignals(True)
            pe.setText(p or "")
            pe.blockSignals(False)
            mua.setText("" if a in ("", None) else f"{float(a):g}")
            musp.setText("" if b in ("", None) else f"{float(b):g}")
        self.changed.emit()

    def set_slot(self, i, path, mua, musp):
        if i == 1:
            self.enable_sec.setChecked(True)
        cur = [self.slot(0), self.slot(1)]
        cur[i] = [path, f"{mua:g}" if isinstance(mua, (int, float)) else str(mua or ""),
                        f"{musp:g}" if isinstance(musp, (int, float)) else str(musp or "")]
        self.set_rows(cur)

    def add_recording(self, path, mua, musp):
        """a phantom just recorded: into the first free slot (or the same file's slot); both full -> the first"""
        cur = [self.slot(i) for i in range(2)]
        same = [i for i in range(2) if cur[i][0] and os.path.abspath(cur[i][0]) == os.path.abspath(path)]
        i = same[0] if same else (0 if not cur[0][0] else (1 if self.enable_sec.isChecked() and not cur[1][0] else 0))
        cur[i] = [path, f"{mua:g}", f"{musp:g}"]
        self.set_rows(cur)
        return i


def _section(title=None):
    """A titled block without its own card (for forms that sit inside a card)."""
    f = QtWidgets.QWidget()
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(0, 4, 0, 0)
    lay.setSpacing(6)
    if title:
        t = QtWidgets.QLabel(title)
        t.setObjectName("sectionTitle")
        lay.addWidget(t)
    return f, lay


class CalForm(QtWidgets.QWidget):
    """Calibration inputs: reference recordings with certified values, optional custom DL, probe geometry, and solver parameters (uses internal HOCAL time-zero)."""
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.w = {}
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)

        s_refs, lay_refs = _section("1 · Reference Phantom Recording(s)")
        self.refs = RefPair()
        self.refs.changed.connect(lambda *_: self.changed.emit())
        lay_refs.addWidget(self.refs)
        v.addWidget(s_refs)

        # Hidden placeholder for backward compatibility with saved state
        self.w["irf"] = PathEdit("file")

        s_dl, lay_dl = _section("2 · Calibration Direct Light (DL)")
        self.diff_cal_dl = QtWidgets.QCheckBox("Use a different DL for calibration (defaults to data DL)")
        self.diff_cal_dl_box = QtWidgets.QWidget()
        dl_b_lay = QtWidgets.QVBoxLayout(self.diff_cal_dl_box)
        dl_b_lay.setContentsMargins(0, 4, 0, 0)
        dl_b_lay.setSpacing(4)
        self.w["cal_dl"] = PathEdit("file", placeholder="Select calibration DL (.csv)… (leave blank to use data DL)")
        self.w["cal_dl"].changed.connect(lambda *_: self.changed.emit())
        dl_b_lay.addWidget(self.w["cal_dl"])
        self.cal_dl_hint = muted("If left blank, the data Direct Light (DL) will be used.")
        dl_b_lay.addWidget(self.cal_dl_hint)
        self.diff_cal_dl_box.setVisible(False)
        self.diff_cal_dl.toggled.connect(self.diff_cal_dl_box.setVisible)
        self.diff_cal_dl.toggled.connect(lambda *_: self.changed.emit())
        lay_dl.addWidget(self.diff_cal_dl)
        lay_dl.addWidget(self.diff_cal_dl_box)
        v.addWidget(s_dl)

        s_geo, lay_geo = _section("3 · Probe Geometry & Refractive Index")
        f = form_layout()
        self.w["rho"] = dspin(1, 60, 18.0, 2, 0.5, "mm")
        self.w["n"] = dspin(1.0, 2.0, 1.41, 3, 0.01)
        self.w["rho"].setToolTip("Source–detector separation ρ of the optical probe (mm). Stored in the calibration file.")
        self.w["n_out"] = dspin(1.0, 2.0, 1.0, 3, 0.01)
        self.w["n"].setToolTip("Refractive index of the reference phantom medium (determines speed of light v = c/n).")
        self.w["n_out"].setToolTip("Refractive index of the external medium at the boundary (air = 1.000, water = 1.333).")
        self.w["n"].valueChanged.connect(lambda val: self.w["n_out"].setMaximum(val))
        f.addRow("Source–detector separation ρ", self.w["rho"])
        f.addRow("Phantom refractive index (n)", self.w["n"])
        f.addRow("External refractive index (n_ext)", self.w["n_out"])
        lay_geo.addLayout(f)
        v.addWidget(s_geo)

        adv = Collapsible("Advanced Solver Parameters")
        F = adv.form
        self.w["bg_model"] = QtWidgets.QComboBox()
        self.w["bg_model"].addItems(["dl", "exp", "flat"])
        self.w["bg_bins"] = QtWidgets.QLineEdit("0-11,70-127")
        self.w["ped_start"] = ispin(0, 127, 13)
        self.w["win_pre"] = ispin(0, 64, 3, "bins")
        self.w["win_post"] = ispin(1, 127, 25, "bins")
        self.w["win_shift"] = QtWidgets.QLineEdit()
        self.w["win_shift"].setPlaceholderText("auto (leave blank for auto-shift)")
        self.w["sig_t"] = dspin(0.1, 1000, 6.0, 1, 1, "ps")
        self.w["sig_i"] = dspin(0.001, 5, 0.03, 3, 0.01)
        self.w["zones"] = QtWidgets.QLineEdit()
        self.w["zones"].setPlaceholderText("all channels (leave blank to use all)")
        self.w["min_signal"] = dspin(0, 1, 0.2, 2, 0.05)
        self.w["extra"] = QtWidgets.QLineEdit()
        self.w["extra"].setPlaceholderText("none (leave blank if none)")
        for lab, k in (("Background model", "bg_model"), ("Background bins", "bg_bins"), ("Pedestal start bin (exp)", "ped_start"),
                       ("Window pre-peak", "win_pre"), ("Window post-peak", "win_post"),
                       ("HOCAL window shift", "win_shift"), ("σ mean time", "sig_t"), ("σ intensity", "sig_i"),
                       ("Active channels (zones)", "zones"), ("Min signal fraction", "min_signal"), ("Extra solver flags", "extra")):
            F.addRow(lab, self.w[k])
        v.addWidget(adv)

    def uses_irf(self):
        return False

    def _tref(self, *_):
        pass

    def get_dl_path(self, fallback_dl=None):
        if self.diff_cal_dl.isChecked() and self.w["cal_dl"].text():
            return self.w["cal_dl"].text()
        return fallback_dl

    def values(self):
        d = widget_values(self.w)
        d["references"] = self.refs.rows()
        d["diff_cal_dl"] = self.diff_cal_dl.isChecked()
        d["enable_sec"] = self.refs.enable_sec.isChecked()
        return d

    def set_values(self, d):
        set_widget_values(self.w, d)
        if "references" in d:
            self.refs.set_rows(d["references"])
        if "diff_cal_dl" in d:
            self.diff_cal_dl.setChecked(bool(d["diff_cal_dl"]))
        if "enable_sec" in d:
            self.refs.enable_sec.setChecked(bool(d["enable_sec"]))

    def prepare(self, work, which=0, session_dl=None):
        """Check the inputs, put reference `which` (0 = primary, 1 = secondary) into the
        folder `work` the way the solver expects, and return the arguments. Raises ValueError with a readable message."""
        w = self.w
        dl = self.get_dl_path(fallback_dl=session_dl)
        if not dl or not os.path.isfile(dl):
            raise ValueError("Please select or record a Direct Light (DL) recording (or enable 'Use a different DL for calibration').")
        r = self.refs.slot(which)
        if not r[0]:
            name = "Primary" if which == 0 else "Secondary"
            raise ValueError(f"Please select the {name} Reference Phantom recording (Step 1).")
        rows = [r]
        truth, names = {}, []
        for p, a, b in rows:
            name = os.path.splitext(os.path.basename(p))[0]
            try:
                a, b = float(a), float(b)
            except ValueError:
                raise ValueError(f"Please enter the certified μa and μs′ values for '{name}' (Step 1).")
            if not os.path.isfile(p):
                raise ValueError(f"Recording not found: {p}")
            if name in truth:
                raise ValueError(f"Two reference files share the name '{name}' — please rename one of the files.")
            shutil.copyfile(p, os.path.join(work, name + ".csv"))
            truth[name] = {"mean_mua": a, "mean_musp": b}
            names.append(name)
        tj = os.path.join(work, "_certified.json")
        with open(tj, "w", encoding="utf-8") as fh:
            json.dump(truth, fh, indent=1)
        a = ["--data", work, "--dl", dl, "--truth", tj, "--refs", ",".join(names), "--hocal"]
        if w["n_out"].value() > w["n"].value():
            raise ValueError("External refractive index (n_ext) cannot exceed the phantom refractive index (n).")
        if w["n_out"].value() != 1.0:
            a += ["--n-out", f"{w['n_out'].value():g}"]
        a += ["--rho", f"{w['rho'].value():g}", "--n", f"{w['n'].value():g}", "--bg-model", w["bg_model"].currentText(),
              "--bg-bins", w["bg_bins"].text().replace(" ", ""), "--ped-start", str(w["ped_start"].value()),
              "--win-pre", str(w["win_pre"].value()), "--win-post", str(w["win_post"].value()),
              "--sig-t", f"{w['sig_t'].value():g}", "--sig-i", f"{w['sig_i'].value():g}",
              "--min-signal", f"{w['min_signal'].value():g}", "--outdir", os.path.join(work, "out")]
        if w["win_shift"].text():
            a += ["--win-shift", w["win_shift"].text()]
        if w["zones"].text():
            a += ["--zones", w["zones"].text().replace(" ", "")]
        if w["extra"].text():
            a += shlex.split(w["extra"].text(), posix=(os.name != "nt"))
        return a


def base(p):
    """File name of a path written on Windows or here."""
    return re.split(r"[\\/]", p or "")[-1]


def cal_summary(cfg):
    s = cfg["settings"]
    sec = cfg.get("second_reference")
    refs = ", ".join(cfg["refs"]) + (f" (secondary: {', '.join(sec.get('refs') or [])})" if sec else "")
    return (
        f"<b>t₀</b> = {cfg['t0_ps']:.1f} ps &nbsp;·&nbsp; "
        f"<b>log K</b> = {cfg['logk']:.3f} &nbsp;·&nbsp; "
        f"<b>ρ</b> = {cfg['rho_mm']:.1f} mm"
        f"<br><span style='color:{C['muted']}'>Ref: {refs} &nbsp;|&nbsp; "
        f"HOCAL time-zero &nbsp;|&nbsp; "
        f"n = {s['N_TISSUE']:g}" + (f" (n_ext = {s.get('N_OUT', 1.0):g})" if s.get("N_OUT", 1.0) != 1.0 else "")
        + f" &nbsp;|&nbsp; {cfg['created'][:10]}</span>"
    )




class TabStack(QtWidgets.QWidget):
    currentChanged = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)
        self.bar_row = QtWidgets.QHBoxLayout()
        self.bar_row.setContentsMargins(0, 0, 0, 0)
        self.bar_row.setSpacing(8)
        self.bar = QtWidgets.QTabBar()
        self.bar.setDrawBase(False)
        self.bar.setExpanding(False)
        self.bar.setUsesScrollButtons(False)
        self.bar.setStyleSheet(f"""
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent;
    color: {C['muted']};
    padding: 8px 16px;
    margin-right: 4px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    font-size: 9.5pt;
}}
QTabBar::tab:selected {{
    color: {C['text']};
    border-bottom: 2px solid {C['accent']};
}}
QTabBar::tab:hover {{
    color: {C['text']};
}}
""")
        self.bar_row.addWidget(self.bar, 0, Qt.AlignVCenter)
        self.bar_row.addStretch(1)
        v.addLayout(self.bar_row)
        self.stack = QtWidgets.QStackedWidget()
        self.stack.setStyleSheet(f"QStackedWidget {{ background: {C['card']}; border: 1px solid {C['border']}; border-radius: 8px; }}")
        v.addWidget(self.stack, 1)
        self.bar.currentChanged.connect(self._tab_changed)

    def _tab_changed(self, i):
        self.stack.setCurrentIndex(i)
        self.currentChanged.emit(i)

    def addTab(self, widget, label):
        self.bar.addTab(label)
        return self.stack.addWidget(widget)

    def setCurrentIndex(self, i):
        self.bar.setCurrentIndex(i)
        self.stack.setCurrentIndex(i)

    def setCurrentWidget(self, w):
        i = self.stack.indexOf(w)
        if i >= 0:
            self.setCurrentIndex(i)

    def setDocumentMode(self, *args):
        pass

    def currentIndex(self):
        return self.stack.currentIndex()

    def currentWidget(self):
        return self.stack.currentWidget()

    def count(self):
        return self.stack.count()




class Kpi(QtWidgets.QFrame):
    def __init__(self, title, color, unit=""):
        super().__init__()
        self.setObjectName("kpi")
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(14, 10, 14, 10)
        v.setSpacing(2)
        self.title_text = title
        self.title_lbl = QtWidgets.QLabel(title)
        self.title_lbl.setObjectName("kpiTitle")
        bar = QtWidgets.QFrame()
        bar.setFixedHeight(3)
        bar.setStyleSheet(f"background: {color}; border-radius: 1px;")
        self.val = QtWidgets.QLabel("—")
        self.val.setObjectName("kpiValue")
        self.sub = QtWidgets.QLabel(unit)
        self.sub.setObjectName("muted")
        v.addWidget(self.title_lbl)
        v.addWidget(self.val)
        v.addWidget(self.sub)
        v.addWidget(bar)

    def set(self, value, sub=None):
        self.val.setText(value)
        if sub is not None:
            self.sub.setText(sub)



