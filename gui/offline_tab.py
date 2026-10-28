"""Offline batch analysis tab and background processing runner."""
import csv
import datetime as dt
import glob
import json
import os
import re
import shlex
import shutil
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import pyqtgraph as pg

import figures as FIG
import tmf_moments_solver as V
import live_engine as L
from gui.styles import (
    C, APP_NAME, _PG_SYM, card, subcard, muted, pill, set_pill, style_plot
)
from gui.persistence import (
    Persist, widget_values, set_widget_values, read_app_json, write_settings
)
from gui.widgets import (
    PathEdit, Collapsible, UnitForm, form_layout, dspin, ispin, unit_box,
    PlotTools, plot_bar, with_bar, RefPair, CalForm, TabStack,
    fill_table, new_table, new_log, scroll, read_csv, base, cal_summary,
    is_recording, read_values_file, _HAVE_PLOTLY, Runner
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = ROOT_DIR
OFFLINE = os.path.join(ROOT_DIR, "offline_engine.py")

SKIP_PREFIX = ("dl", "irf")
SKIP_SUFFIX = ("_live_results.csv", "_timeseries.csv", "_results.csv", "compare_results.csv")




class OfflineTab(QtWidgets.QWidget):
    request_cal_tab = QtCore.Signal()

    def __init__(self, persist):
        super().__init__()
        self.persist = persist
        self.cfg = None
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(16, 14, 16, 16)
        h.setSpacing(16)
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(12)
        self.w = {}

        # 1. Sample Recordings
        c, lay = card("1 · Sample Recordings")
        self.files = QtWidgets.QListWidget()
        self.files.setMinimumHeight(145)
        self.files.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        lay.addWidget(self.files)
        hb = QtWidgets.QHBoxLayout()
        b1 = QtWidgets.QPushButton("Add Files…")
        b1.clicked.connect(self.add_files)
        b2 = QtWidgets.QPushButton("Add Folder…")
        b2.clicked.connect(self.add_folder)
        b2.setToolTip("Adds all sample recordings in the folder (automatically skips DL and result files).")
        b3 = QtWidgets.QPushButton("Remove")
        b3.setObjectName("small")
        b3.clicked.connect(self.remove_files)
        b4 = QtWidgets.QPushButton("Clear")
        b4.setObjectName("small")
        b4.clicked.connect(self.files.clear)
        hb.addWidget(b1)
        hb.addWidget(b2)
        hb.addStretch(1)
        hb.addWidget(b3)
        hb.addWidget(b4)
        lay.addLayout(hb)
        lv.addWidget(c)

        # 2. Calibration & Session Overrides
        c, lay = card("2 · Calibration")
        cal_hdr = QtWidgets.QHBoxLayout()
        b_new_cal = QtWidgets.QPushButton("New Calibration →")
        b_new_cal.setObjectName("ghost")
        b_new_cal.setToolTip("Switch to Live Acquisition to compute a new calibration from reference recordings")
        b_new_cal.clicked.connect(self.request_cal_tab.emit)
        cal_hdr.addWidget(b_new_cal)
        cal_hdr.addStretch(1)
        lay.addLayout(cal_hdr)

        self.w["cal"] = PathEdit("file", "Calibration (*.json);;All files (*)", "Select calibration (.json)…")
        self.sc_cal, sclay = subcard()
        self.cal_info = QtWidgets.QLabel("")
        self.cal_info.setWordWrap(True)
        self.cal_info.setTextFormat(Qt.RichText)
        self.cal_info.setObjectName("muted")
        sclay.addWidget(self.cal_info)
        lay.addWidget(self.w["cal"])
        lay.addWidget(self.sc_cal)
        self.sc_cal.setVisible(False)

        self.sess_drawer = Collapsible("Session Direct Light (DL) Override")
        self.w["irf"] = PathEdit("file")  # hidden placeholder for backward compatibility
        self.w["sess_dl"] = PathEdit("file", placeholder="Session DL (.csv)… (leave blank to use calibration's)")
        self.w["sess_dl"].setToolTip("Subtract background using this session's DL instead of the calibration's DL (t₀ and K remain unchanged).")
        self.sess_drawer.form.addRow("Session DL", self.w["sess_dl"])
        self.sess_hint = muted("Leave blank to use the Direct Light stored in the calibration file.")
        self.sess_drawer.form.addRow(self.sess_hint)
        lay.addWidget(self.sess_drawer)
        lv.addWidget(c)

        # 3. Fitting Parameters
        c, lay = card("3 · Fitting Parameters")
        f = form_layout()
        self.w["n"] = dspin(1.0, 2.0, 1.41, 3, 0.01)
        self.w["n_out"] = dspin(1.0, 2.0, 1.0, 3, 0.01)
        self.w["n"].setToolTip("Sample refractive index n (overridden per sample if an 'n' column is present in the validation CSV).")
        self.w["n_out"].setToolTip("External medium refractive index n_ext at the boundary (air = 1.000, water = 1.333).")
        self.w["n"].valueChanged.connect(lambda v: self.w["n_out"].setMaximum(v))
        f.addRow("Sample refractive index (n)", self.w["n"])
        f.addRow("External index (n_ext)", self.w["n_out"])
        self.w["mode"] = QtWidgets.QComboBox()
        self.w["mode"].addItems(["μa and μs′ (free fit)", "μa only (fix μs′)", "μs′ only (fix μa)"])
        f.addRow("Fitted parameters", self.w["mode"])
        self.w["fix_val"] = dspin(0.001, 50, 10, 3, 0.1, "cm⁻¹")
        self.fix_lbl = QtWidgets.QLabel("Fixed μs′")
        f.addRow(self.fix_lbl, self.w["fix_val"])
        self.w["win_pre"] = ispin(1, 128, 12)
        self.w["win_post"] = ispin(1, 128, 40)
        self.w["win_pre"].setToolTip("Integration window before reference peak (WIN_PRE bins)")
        self.w["win_post"].setToolTip("Integration window after reference peak (WIN_POST bins)")
        f.addRow("Fit window pre-peak", unit_box(self.w["win_pre"], "bins"))
        f.addRow("Fit window post-peak", unit_box(self.w["win_post"], "bins"))
        lay.addLayout(f)
        self.w["mode"].currentIndexChanged.connect(self._mode_changed)
        lv.addWidget(c)

        # 4. Plotting Options
        c, lay = card("4 · Plotting Options")
        self.w["ts"] = QtWidgets.QCheckBox("Plot time series")
        self.w["ts"].setChecked(True)
        lay.addWidget(self.w["ts"], alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignAbsolute)

        self.roll_box = QtWidgets.QWidget()
        hr = QtWidgets.QHBoxLayout(self.roll_box)
        hr.setContentsMargins(18, 0, 0, 0)
        self.w["roll_on"] = QtWidgets.QCheckBox("Moving Average Window (s):")
        self.w["roll_s"] = dspin(0.1, 3600, 5, 1, 1)
        self.w["roll_s"].setFixedWidth(90)
        hr.addWidget(self.w["roll_on"])
        hr.addWidget(self.w["roll_s"])
        hr.addStretch(1)
        lay.addWidget(self.roll_box)

        self.w["cmp_on"] = QtWidgets.QCheckBox("Plot against expected values")
        lay.addWidget(self.w["cmp_on"], alignment=QtCore.Qt.AlignLeft | QtCore.Qt.AlignAbsolute)

        self.cmp_box = QtWidgets.QWidget()
        cv = QtWidgets.QVBoxLayout(self.cmp_box)
        cv.setContentsMargins(18, 0, 0, 0)
        cv.setSpacing(6)
        self.w["compare"] = PathEdit("file", "CSV (*.csv);;All files (*)", "Select expected values (.csv)…")
        cv.addWidget(self.w["compare"])

        cmp_sub = QtWidgets.QHBoxLayout()
        cmp_sub.setSpacing(8)
        cmp_hint = muted("CSV format: sample, μa (cm⁻¹), μs′ (cm⁻¹)")
        b_tmpl = QtWidgets.QPushButton("Download CSV Template…")
        b_tmpl.setObjectName("ghost")
        b_tmpl.setToolTip("Save a sample expected values CSV template with standard column headers")
        b_tmpl.clicked.connect(self.download_expected_template)
        cmp_sub.addWidget(cmp_hint, 1)
        cmp_sub.addWidget(b_tmpl)
        cv.addLayout(cmp_sub)
        lay.addWidget(self.cmp_box)

        for a in ("ts", "roll_on", "cmp_on"):
            self.w[a].toggled.connect(self._enable)
        self._enable()
        lv.addWidget(c)

        lv.addStretch(1)

        outer = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(10)
        ov.addWidget(scroll(left, 460), 1)
        hb_run = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run Analysis")
        self.run_btn.setObjectName("primary")
        self.run_btn.clicked.connect(self.run)
        self.stop_btn = QtWidgets.QPushButton("Stop Analysis")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop)
        hb_run.addWidget(self.run_btn, 2)
        hb_run.addWidget(self.stop_btn, 1)
        ov.addLayout(hb_run)
        outer.setFixedWidth(460)
        h.addWidget(outer)

        # ---------- right: results & figures
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        self.tabs = TabStack()
        self.state = pill("", C["muted"])
        self.state.setVisible(False)
        self.tabs.bar_row.addWidget(self.state, 0, Qt.AlignVCenter)

        self.csv_btn = QtWidgets.QPushButton("Export Table (.csv)…")
        self.csv_btn.setToolTip("Export the summary results table (and linearity fits if validation is enabled) as CSV.")
        self.csv_btn.clicked.connect(self.download_csv)
        self.csv_btn.setEnabled(False)

        self.dl_all = QtWidgets.QPushButton("Export All Figures && Data…")
        self.dl_all.setObjectName("primary")
        self.dl_all.setToolTip("Export all publication figures (PNG + interactive HTML), summary tables, and time-series data into a timestamped folder.")
        self.dl_all.clicked.connect(self.download_all)
        self.dl_all.setEnabled(False)

        self.tabs.bar_row.addWidget(self.csv_btn, 0, Qt.AlignVCenter)
        self.tabs.bar_row.addWidget(self.dl_all, 0, Qt.AlignVCenter)
        self.temp_out = None
        self.tabs.setDocumentMode(True)
        self.table = new_table()
        rt = QtWidgets.QWidget()
        rtl = QtWidgets.QVBoxLayout(rt)
        rtl.setContentsMargins(8, 8, 8, 8)
        rtl.setSpacing(6)
        rtl.addWidget(self.table, 1)
        self.tabs.addTab(rt, "Summary Table")

        ts = QtWidgets.QWidget()
        tl = QtWidgets.QHBoxLayout(ts)
        tl.setContentsMargins(8, 8, 8, 8)
        tl.setSpacing(10)
        side = QtWidgets.QVBoxLayout()
        self.ts_list = QtWidgets.QListWidget()
        self.ts_list.setMaximumWidth(250)
        self.ts_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.ts_list.setToolTip("Select a figure to view. Ctrl/Shift-click multiple recordings to overlay their time series.")
        self.ts_list.itemSelectionChanged.connect(self.plot_ts)
        self.ts_roll = dspin(0, 3600, 0, 1, 1, "s")
        self.ts_roll.setToolTip("Moving average window for time-series display (0 = unsmoothed)")
        self.ts_roll.valueChanged.connect(self.plot_ts)
        self.ts_err = QtWidgets.QCheckBox("Show ±1 SD error bars")
        self.ts_err.toggled.connect(self.plot_ts)
        side.addWidget(self.ts_list, 1)
        rl = QtWidgets.QHBoxLayout()
        rl.addWidget(QtWidgets.QLabel("Moving avg"))
        rl.addWidget(unit_box(self.ts_roll), 1)
        side.addLayout(rl)
        side.addWidget(self.ts_err)
        tl.addLayout(side)

        self.pgw = pg.GraphicsLayoutWidget()
        self.pgw.setBackground("w")
        self.pg_tools = PlotTools(self.pgw, [], "offline_plot", png=lambda: self.download_one("png"),
                                  html=lambda: self.download_one("html"))
        tl.addWidget(with_bar(plot_bar([], self.pg_tools), self.pgw), 1)
        self.tabs.addTab(ts, "Plots")

        self.cmp_rows = []
        self.log = new_log()
        self.tabs.addTab(self.log, "Analysis Log")
        rv.addWidget(self.tabs, 1)
        h.addWidget(right, 1)

        self.runner = Runner(self.log)
        self.runner.finished.connect(self.done)
        self.out = None
        persist.load("off2", self.w)
        try:
            for p in json.loads(persist.s.value("off2/files", "[]")):
                self._add(p)
        except Exception:
            pass
        self.w["cal"].changed.connect(self.load_cal)
        if self.w["cal"].text():
            keep = widget_values(self.w)
            self.load_cal(self.w["cal"].text())
            set_widget_values(self.w, {k: keep[k] for k in ("n", "n_out", "irf") if k in keep})
            self._tz_note()
        if self.w["irf"].text() or self.w["sess_dl"].text():
            self.sess_drawer.btn.setChecked(True)
        self._alt_note()
        self._mode_changed(self.w["mode"].currentIndex(), keep=True)
        self._enable()

    # ---- inputs
    def _add(self, p):
        p = os.path.normpath(p)
        if any(self.files.item(i).data(Qt.UserRole) == p for i in range(self.files.count())):
            return
        it = QtWidgets.QListWidgetItem(f"{os.path.basename(p)}    ·  {os.path.basename(os.path.dirname(p))}")
        it.setData(Qt.UserRole, p)
        it.setToolTip(p)
        self.files.addItem(it)

    def file_paths(self):
        return [self.files.item(i).data(Qt.UserRole) for i in range(self.files.count())]

    def _start_dir(self):
        ps = self.file_paths()
        return os.path.dirname(ps[-1]) if ps else ""

    def add_files(self):
        ps, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Add Sample Recordings", self._start_dir(), "CSV files (*.csv)")
        for p in ps:
            self._add(p)

    def add_folder(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Add All Recordings in Folder", self._start_dir())
        if not d:
            return
        for fn in sorted(os.listdir(d), key=V.natural_key):
            low = fn.lower()
            if low.endswith(".csv") and not low.startswith(SKIP_PREFIX) and not low.endswith(SKIP_SUFFIX):
                self._add(os.path.join(d, fn))

    def remove_files(self):
        for it in self.files.selectedItems():
            self.files.takeItem(self.files.row(it))

    def load_cal(self, path):
        self.cfg = None
        if not path:
            self.cal_info.setText("")
            self.sc_cal.setVisible(False)
            return
        try:
            self.cfg = L.load_calibration(path)
        except Exception as e:
            self.cal_info.setText(f"<span style='color:{C['bad']}'>Could not read calibration: {e}</span>")
            self.sc_cal.setVisible(True)
            return
        self.cal_info.setText(cal_summary(self.cfg))
        self.sc_cal.setVisible(True)
        if not self.cfg.get("no_settings"):
            self.w["n"].setValue(self.cfg["settings"]["N_TISSUE"])
            self.w["n_out"].setValue(self.cfg["settings"].get("N_OUT", 1.0))
            if "WIN_PRE" in self.cfg["settings"]:
                self.w["win_pre"].setValue(int(self.cfg["settings"]["WIN_PRE"]))
            if "WIN_POST" in self.cfg["settings"]:
                self.w["win_post"].setValue(int(self.cfg["settings"]["WIN_POST"]))
        self.w["irf"].setText("")

    def _alt_note(self, *_):
        pass

    def _tz_note(self, *_):
        pass

    def _mode_changed(self, i, keep=False):
        if not keep:
            self.w["fix_val"].setValue(10.0 if i == 1 else 0.1)
        self.fix_lbl.setText("Fixed μs′" if i == 1 else "Fixed μa")
        for wdg in (self.fix_lbl, self.w["fix_val"].unit_container):
            wdg.setVisible(i != 0)

    def _enable(self, *_):
        ts = self.w["ts"].isChecked()
        if hasattr(self, "roll_box"):
            self.roll_box.setVisible(ts)
        self.w["roll_on"].setEnabled(ts)
        self.w["roll_s"].setEnabled(ts and self.w["roll_on"].isChecked())
        cmp_checked = self.w["cmp_on"].isChecked()
        if hasattr(self, "cmp_box"):
            self.cmp_box.setVisible(cmp_checked)
        else:
            self.w["compare"].setEnabled(cmp_checked)
            self.w["compare"].setVisible(cmp_checked)

    def download_expected_template(self):
        dest, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Expected Values Template", "expected_values_template.csv", "CSV files (*.csv)"
        )
        if dest:
            try:
                dest = dest if dest.lower().endswith(".csv") else dest + ".csv"
                with open(dest, "w", newline="", encoding="utf-8") as f:
                    f.write("sample,mua,musp\nSample_01,0.080,8.50\nSample_02,0.120,12.00\nSample_03,0.250,15.50\n")
                set_pill(self.state, f"Template saved: {os.path.basename(dest)}", C["ok"])
                self.state.setVisible(True)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not save template: {e}")

    def save_state(self):
        self.persist.save("off2", self.w)
        self.persist.s.setValue("off2/files", json.dumps(self.file_paths()))

    # ---- run
    def run(self):
        if self.runner.running():
            return
        files = self.file_paths()
        msg = None
        if not files:
            msg = "Please add at least one recording to analyse (Step 1)."
        elif self.cfg is None:
            msg = "Please select a calibration (.json) file (Step 2)."
        elif self.w["sess_dl"].text() and not os.path.isfile(self.w["sess_dl"].text()):
            msg = "Session Direct Light (DL) recording not found."
        elif self.w["irf"].text() and not os.path.isfile(self.w["irf"].text()):
            msg = "Session IRF recording not found (clear the field to use the calibration's time-zero)."
        elif self.w["cmp_on"].isChecked() and not os.path.isfile(self.w["compare"].text()):
            msg = "Please select the CSV file of expected phantom values (or uncheck validation)."
        if msg:
            QtWidgets.QMessageBox.warning(self, APP_NAME, msg)
            return
        self._drop_temp()
        import tempfile
        out = self.temp_out = tempfile.mkdtemp(prefix="ptof_offline_")
        mode = self.w["mode"].currentIndex()
        job = {"calibration": self.w["cal"].text(), "files": files,
               "fit": {"mode": ("both", "fix_musp", "fix_mua")[mode], "fix_value": self.w["fix_val"].value(),
                       "fix_from": "both",
                       "n": self.w["n"].value(), "n_out": min(self.w["n_out"].value(), self.w["n"].value()),
                       "win_pre": self.w["win_pre"].value(), "win_post": self.w["win_post"].value()},
               "outputs": {"timeseries": self.w["ts"].isChecked(),
                           "rolling_s": self.w["roll_s"].value() if (self.w["ts"].isChecked() and self.w["roll_on"].isChecked()) else 0,
                           "compare": self.w["compare"].text() if self.w["cmp_on"].isChecked() else None},
               "session_dl": self.w["sess_dl"].text() or None,
               "time_zero": {"mode": "hocal", "irf": None},
               "outdir": out, "created": dt.datetime.now().isoformat(timespec="seconds")}
        os.makedirs(out, exist_ok=True)
        jp = os.path.join(out, "offline_job.json")
        with open(jp, "w", encoding="utf-8") as fh:
            json.dump(job, fh, indent=1, ensure_ascii=False)
        self.save_state()
        self.out = out
        self.log.clear()
        self.tabs.setCurrentWidget(self.log)
        set_pill(self.state, f"Analysing {len(files)} recording(s)…", C["warn"])
        self.state.setVisible(True)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.ts_roll.setValue(job["outputs"]["rolling_s"])
        self.runner.start([OFFLINE, jp])

    def stop(self):
        if self.runner.running():
            set_pill(self.state, "Analysis Stopped", C["warn"])
            self.state.setVisible(True)
            self.log.appendPlainText("\n[Analysis stopped by user]\n")
            self.runner.kill()
            self.stop_btn.setEnabled(False)
            self.run_btn.setEnabled(True)

    def done(self, code):
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code != 0:
            if self.state.text() != "Analysis Stopped":
                set_pill(self.state, "Failed — see Analysis Log", C["bad"])
                self.state.setVisible(True)
            return
        set_pill(self.state, "Analysis Complete", C["ok"])
        self.state.setVisible(True)
        res = os.path.join(self.out, "offline_results.csv")
        self.csv_btn.setEnabled(os.path.isfile(res))
        if os.path.isfile(res):
            h, rows = read_csv(res)
            show = [("sample", "Sample"), ("mua_cm", "μa (cm⁻¹)"), ("mua_sd", "± SD"), ("musp_cm", "μs′ (cm⁻¹)"), ("musp_sd", "± SD"),
                    ("fixed", "Fixed"), ("calibration", "Calibration"), ("mua_expected", "μa Expected"), ("mua_err_pct", "μa Error (%)"),
                    ("musp_expected", "μs′ Expected"), ("musp_err_pct", "μs′ Error (%)")]
            has_cmp = any(r[h.index("mua_expected")] for r in rows)
            show = [s for s in show if has_cmp or "expected" not in s[0] and "err" not in s[0]]
            if not any(r[h.index("fixed")] for r in rows):
                show = [s for s in show if s[0] != "fixed"]
            cals = {r[h.index("calibration")] for r in rows}
            if len(cals) <= 1 and not any(c.endswith("(circular)") for c in cals):
                show = [s for s in show if s[0] != "calibration"]
            fill_table(self.table, [s[1] for s in show], [[r[h.index(s[0])] for s in show] for r in rows])
            ci = h.index("calibration")
            if "calibration" in [x[0] for x in show]:
                self.table.horizontalHeader().setSectionResizeMode([x[0] for x in show].index("calibration"),
                                                                   QtWidgets.QHeaderView.ResizeToContents)
            for i, r in enumerate(rows):
                if r[ci].endswith("(circular)"):
                    for j in range(self.table.columnCount()):
                        self.table.item(i, j).setForeground(QtGui.QColor(C["muted"]))
                        self.table.item(i, j).setToolTip("Primary reference phantom of the calibration — fitted with itself (circular)")
            self.tabs.setCurrentIndex(0)
        self.load_plots()

    # ---- Figures & Time Series tab
    def load_plots(self):
        """Fill the plot list from the results folder: summary plots first, then one entry per recording."""
        self.ts_list.blockSignals(True)
        self.ts_list.clear()
        self.cmp_rows = []
        res = os.path.join(self.out, "offline_results.csv")
        if os.path.isfile(res):
            h, rows = read_csv(res)
            ix = {k: h.index(k) for k in h}
            for r in sorted(rows, key=lambda r: V.natural_key(r[ix["sample"]])):
                if not r[ix["mua_expected"]] or r[ix["calibration"]].endswith("(circular)"):
                    continue
                v = tuple(float(r[ix[k]]) for k in ("mua_cm", "mua_sd", "musp_cm", "musp_sd", "mua_expected",
                                                    "musp_expected"))
                self.cmp_rows.append((r[ix["sample"]],) + v[:4] + v[4:])
        ts = sorted(glob.glob(os.path.join(self.out, "timeseries", "*_timeseries.csv")), key=V.natural_key)

        def add(text, data, tip=""):
            it = QtWidgets.QListWidgetItem(text)
            it.setData(Qt.UserRole, data)
            it.setToolTip(tip)
            self.ts_list.addItem(it)
            return it

        def head(text):
            it = QtWidgets.QListWidgetItem(text.upper())
            it.setFlags(Qt.NoItemFlags)
            it.setForeground(QtGui.QColor(C["accent"]))
            f = it.font()
            f.setPointSizeF(f.pointSizeF() * 0.85)
            f.setBold(True)
            it.setFont(f)
            self.ts_list.addItem(it)

        if self.cmp_rows:
            head("Validation")
            add("Recovered vs. Expected", ("compare",), "Linearity & accuracy across all samples with certified values (y = x and ±10 % band)")
        if ts:
            head("Time Series")
            if len(ts) > 1:
                add("All Recordings (Overlay)", ("ts_all",))
            for f in ts:
                add(os.path.basename(f)[:-15], ("ts", f))
        self.ts_list.blockSignals(False)
        self.dl_all.setEnabled(self.ts_list.count() > 0)
        first = next((self.ts_list.item(i) for i in range(self.ts_list.count())
                      if self.ts_list.item(i).flags() & Qt.ItemIsSelectable), None)
        if first:
            first.setSelected(True)
        else:
            self.plot_ts()

    @staticmethod
    def _light(p, ylab="", xlab="", title=""):
        ink = FIG.INK
        for ax in ("left", "bottom"):
            a = p.getAxis(ax)
            a.setPen(pg.mkPen(ink, width=1))
            a.setTextPen(ink)
            a.setStyle(tickLength=5)
            a.enableAutoSIPrefix(False)
        p.getAxis("left").setGrid(40)
        if ylab:
            p.setLabel("left", ylab, color=ink)
        if xlab:
            p.setLabel("bottom", xlab, color=ink)
        if title:
            p.setTitle(f"<span style='color:{ink}; font-size:12.5pt'>{title}</span>", justify="left")
        return p

    @staticmethod
    def _legend(p, cols=1):
        lg = p.addLegend(offset=(12, 6), labelTextColor=FIG.INK, brush=None, pen=None, colCount=cols,
                         labelTextSize="9.5pt")
        return lg

    @staticmethod
    def _sym(mk):
        return _PG_SYM.get(mk, "o")

    def _pts_item(self, p, x, y, col, mk, size=10, line=True, width=2.4, name=None, dash=False):
        pen = pg.mkPen(col, width=width, style=Qt.DashLine if dash else Qt.SolidLine) if line else None
        sym = self._sym(mk) if mk else ("o" if line else None)
        return p.plot(x, y, pen=pen, symbol=sym, symbolSize=size, symbolBrush=col,
                      symbolPen=pg.mkPen("w", width=1.2), name=name, connect="finite")

    def _errbars(self, p, x, y, sd, col):
        if self.ts_err.isChecked():
            p.addItem(pg.ErrorBarItem(x=np.asarray(x, float), y=np.asarray(y, float), height=2 * np.asarray(sd, float),
                                      pen=pg.mkPen(col, width=1.2), beam=0.08))

    def plot_ts(self):
        gw = self.pgw
        gw.clear()
        sel = [it.data(Qt.UserRole) for it in self.ts_list.selectedItems() if it.data(Qt.UserRole)]
        if not sel:
            self.pg_tools.plots = []
            return
        kinds = {d[0] for d in sel}
        if kinds == {"ts"} and len(sel) > 1:
            plots = self._draw_ts([d[1] for d in sel])
        elif sel[-1][0] in ("ts", "ts_all"):
            files = [d[1] for d in sel if d[0] == "ts"] if sel[-1][0] == "ts" else \
                sorted(glob.glob(os.path.join(self.out, "timeseries", "*_timeseries.csv")), key=V.natural_key)
            plots = self._draw_ts(files if sel[-1][0] == "ts_all" else [sel[-1][1]])
        elif sel[-1][0] == "compare":
            plots = self._draw_compare()
        else:
            plots = []
        self.pg_tools.plots = plots

    def _expected_of(self, name):
        for r in self.cmp_rows:
            if r[0] == name:
                return r[5], r[6]
        return None

    def _draw_ts(self, files):
        p1 = self._light(self.pgw.addPlot(row=0, col=0), FIG.MUA_LAB)
        p2 = self._light(self.pgw.addPlot(row=1, col=0), FIG.MUSP_LAB, "time (s)")
        p2.setXLink(p1)
        roll = self.ts_roll.value()
        single = len(files) == 1
        if single:
            name = os.path.basename(files[0])[:-15]
            roll_tag = f" ({roll:g} s moving average)" if roll else ""
            p1.setTitle(f"<span style='color:{FIG.INK}; font-size:12.5pt'>{name}{roll_tag}</span>", justify="left")
            if roll:
                self._legend(p1)
            exp = None
        else:
            p1.setTitle(f"<span style='color:{FIG.INK}; font-size:12.5pt'>{len(files)} recordings"
                        + (f" ({roll:g} s moving average)" if roll else "") + "</span>", justify="left")
            self._legend(p1, 1 if len(files) <= 8 else 2)
            exp = None
        for i, f in enumerate(files):
            try:
                d = np.atleast_1d(np.genfromtxt(f, delimiter=",", names=True))
            except Exception:
                continue
            t = d["t_s"]
            name = os.path.basename(f)[:-15]
            w = FIG.rolling_window(t, roll)
            for k, (p, key, _sd) in enumerate(((p1, "mua_cm", "mua_sd"), (p2, "musp_cm", "musp_sd"))):
                y = d[key]
                if single:
                    col = FIG.MUA_COL if k == 0 else FIG.MUSP_COL
                    if w > 1:
                        c = QtGui.QColor(col)
                        c.setAlpha(80)
                        p.plot(t, y, pen=pg.mkPen(c, width=0.8), symbol="o", symbolSize=4, symbolBrush=c, symbolPen=None,
                               name="every frame" if k == 0 else None)
                        self._pts_item(p, t, V.moving_average(y, w), col, "o", size=6, width=2.4,
                                       name=f"moving avg ({roll:g} s)" if k == 0 else None)
                    else:
                        self._pts_item(p, t, y, col, "o", size=6, width=1.4)
                else:
                    mk, col = FIG.overview_style(name, i)
                    yy = V.moving_average(y, w) if w > 1 else y
                    self._pts_item(p, t, yy, col, mk, size=7, width=1.6, name=name if k == 0 else None)
        return [("μa", p1), ("μs′", p2)]

    def _draw_compare(self):
        data = FIG.all_linear(self.cmp_rows, self.ts_err.isChecked())
        out = []
        for c, which, lab in ((0, "mua", "μa"), (1, "musp", "μs′")):
            lim, _fit, pts = data[which]
            p = self._light(self.pgw.addPlot(row=0, col=c), f"recovered {lab} (cm⁻¹)", f"expected {lab} (cm⁻¹)",
                            f"{lab}: recovered vs expected")
            p.getAxis("bottom").setGrid(40)
            self._legend(p)
            xx = np.array([0.0, lim])
            clear = pg.mkPen(0, 0, 0, 0)
            lo = p.plot(xx, 0.9 * xx, pen=clear)
            hi = p.plot(xx, 1.1 * xx, pen=clear)
            band = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush("#dcefe3"))
            band.setZValue(-1)
            p.addItem(band)
            p.plot([0, 0], [0, 0], pen=None, fillLevel=0, brush=pg.mkBrush("#cfe9d8"), name="±10 %")
            p.plot(xx, xx, pen=pg.mkPen("#777777", width=1.6, style=Qt.DashLine), name="y = x (expected)")
            for name, mk, col, x, y, sd in pts:
                self._pts_item(p, [x], [y], col, mk, size=11, line=False)
                self._errbars(p, [x], [y], [sd], col)
                t = pg.TextItem(name, color="#444444", anchor=(-0.15, 1.0))
                t.setPos(x, y)
                p.addItem(t)
            p.setXRange(0, lim, padding=0.02)
            p.setYRange(0, lim, padding=0.02)
            out.append((lab, p))
        return out

    # ---- exports: PNG (paper style, figures.py) + HTML (interactive, plotly)
    def _all_ts_files(self):
        return sorted(glob.glob(os.path.join(self.out, "timeseries", "*_timeseries.csv")), key=V.natural_key)

    def _spec(self, data_list):
        if not data_list:
            return None
        kinds = {d[0] for d in data_list}
        last = data_list[-1]
        if kinds == {"ts"} and len(data_list) > 1:
            files = [d[1] for d in data_list]
            return "ts", files, "timeseries_" + "_".join(os.path.basename(f)[:-15] for f in files[:6])
        if last[0] == "ts":
            return "ts", [last[1]], os.path.basename(last[1])[:-15] + "_timeseries"
        if last[0] == "ts_all":
            return "ts", self._all_ts_files(), "timeseries_all_recordings"
        if last[0] == "compare":
            return "compare", None, "recovered_vs_expected"
        return None

    def _export(self, spec, base, plotlyjs=True, png=True, html=True):
        kind, arg, _ = spec
        html = html and _HAVE_PLOTLY
        err = self.ts_err.isChecked()
        if kind == "ts":
            all_ts = {}
            for f in arg:
                try:
                    A = np.atleast_2d(np.genfromtxt(f, delimiter=",", skip_header=1))[:, :8]
                except Exception:
                    continue
                all_ts[os.path.basename(f)[:-15]] = A
            if not all_ts:
                raise RuntimeError("no time series data")
            roll = self.ts_roll.value()
            if len(all_ts) == 1:
                name, A = next(iter(all_ts.items()))
                if png:
                    FIG.timeseries(A, name, base + ".png", None, FIG.rolling_window(A[:, 1], roll), window_s=roll)
                if html:
                    FIG.html_timeseries(all_ts, base + ".html", roll, None, plotlyjs)
            else:
                if png:
                    FIG.overview(all_ts, base + ".png", roll)
                if html:
                    FIG.html_timeseries(all_ts, base + ".html", roll, None, plotlyjs)
        elif kind == "compare":
            if png:
                FIG.compare(self.cmp_rows, base + ".png", errbars=err)
            if html:
                FIG.html_compare(self.cmp_rows, base + ".html", err, plotlyjs)

    def _dl_dir(self):
        st = QtCore.QSettings("FAST-Biophotonics", "pToF-Moments")
        d = st.value("plot_dir", "")
        if not d or not os.path.isdir(d):
            d = self._start_dir() or os.path.expanduser("~")
        return st, d

    def _no_plotly_note(self):
        if not _HAVE_PLOTLY:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "Interactive HTML export requires plotly. Install it once:\n\n    pip install plotly\n\n"
                "then restart the application. (PNG figures were saved.)"
            )

    def download_one(self, fmt="png"):
        spec = self._spec([it.data(Qt.UserRole) for it in self.ts_list.selectedItems() if it.data(Qt.UserRole)])
        if not spec:
            return
        if fmt == "html" and not _HAVE_PLOTLY:
            self._no_plotly_note()
            return
        st, d = self._dl_dir()
        flt = "PNG image (*.png)" if fmt == "png" else "Interactive HTML (*.html)"
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Figure as " + ("PNG" if fmt == "png" else "Interactive HTML"),
            os.path.join(d, f"{spec[2]}.{fmt}"), flt)
        if not p:
            return
        base_path = os.path.splitext(p)[0]
        QtWidgets.QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._export(spec, base_path, True, png=fmt == "png", html=fmt == "html")
        except Exception as e:
            QtWidgets.QApplication.restoreOverrideCursor()
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not export figure: {e}")
            return
        QtWidgets.QApplication.restoreOverrideCursor()
        st.setValue("plot_dir", os.path.dirname(p))
        set_pill(self.state, f"Exported {os.path.basename(base_path)}.{fmt}", C["ok"])

    def download_all(self):
        items = [self.ts_list.item(i).data(Qt.UserRole) for i in range(self.ts_list.count())]
        specs = [self._spec([d]) for d in items if d]
        if not specs:
            return
        st, d = self._dl_dir()
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Destination Folder for Export", d)
        if not d:
            return
        dest = os.path.join(d, f"results_{dt.datetime.now():%Y%m%d_%H%M%S}")
        os.makedirs(dest, exist_ok=True)
        prog = QtWidgets.QProgressDialog("Exporting figures and tables…", "Cancel", 0, len(specs), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.setMinimumDuration(300)
        bad = []
        for i, spec in enumerate(specs):
            prog.setValue(i)
            QtWidgets.QApplication.processEvents()
            if prog.wasCanceled():
                break
            try:
                self._export(spec, os.path.join(dest, spec[2]), "directory")
            except Exception as e:
                bad.append(f"{spec[2]}: {e}")
        prog.setValue(len(specs))
        for fn in ("offline_results.csv", "linearity_fits.csv", "offline_job.json"):
            if os.path.isfile(os.path.join(self.out, fn)):
                shutil.copyfile(os.path.join(self.out, fn), os.path.join(dest, fn))
        tsd = os.path.join(self.out, "timeseries")
        if os.path.isdir(tsd):
            shutil.copytree(tsd, os.path.join(dest, "timeseries_data"))
        st.setValue("plot_dir", d)
        set_pill(self.state, f"Exported {len(specs) - len(bad)} figures + tables to {os.path.basename(dest)}", C["ok"])
        if bad:
            QtWidgets.QMessageBox.warning(self, APP_NAME, "Some figures could not be exported:\n" + "\n".join(bad))
        self._no_plotly_note()
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(dest))

    def _drop_temp(self):
        if getattr(self, "temp_out", None):
            shutil.rmtree(self.temp_out, ignore_errors=True)

    def download_csv(self):
        res = os.path.join(self.out or "", "offline_results.csv")
        if not os.path.isfile(res):
            return
        st, d = self._dl_dir()
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Summary Results Table", os.path.join(d, f"offline_results_{dt.datetime.now():%Y%m%d_%H%M%S}.csv"),
            "CSV (*.csv)")
        if not p:
            return
        if not p.lower().endswith(".csv"):
            p += ".csv"
        shutil.copyfile(res, p)
        lin = os.path.join(self.out, "linearity_fits.csv")
        extra = ""
        if os.path.isfile(lin):
            lp = os.path.splitext(p)[0] + "_linearity_fits.csv"
            shutil.copyfile(lin, lp)
            extra = " + linearity fits"
        st.setValue("plot_dir", os.path.dirname(p))
        set_pill(self.state, f"Exported {os.path.basename(p)}{extra}", C["ok"])



