"""Live acquisition, real-time fitting, and pop-up optical window."""
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
import live_engine as L
import tmf_moments_solver as V
from gui.styles import (
    C, APP_NAME, _PG_SYM, card, subcard, muted, rgba, pill, set_pill, style_plot, CountAxisItem
)
from gui.persistence import (
    Persist, widget_values, set_widget_values, read_app_json, write_settings
)
from gui.widgets import (
    PathEdit, Collapsible, UnitForm, form_layout, dspin, ispin, unit_box,
    PlotTools, plot_bar, with_bar, HistControls, LimitsDialog, base,
    CalForm, cal_summary, new_log, scroll, hist_x, hist_y, Runner, Kpi, ChannelPicker
)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HERE = ROOT_DIR
CAL_CLI = os.path.join(ROOT_DIR, "calibrate_cli.py")


class LiveOpticalWindow(QtWidgets.QWidget):
    """Pop-up window for live optical properties (μa and μs′) time-series graphs."""
    dock_requested = QtCore.Signal()
    closed = QtCore.Signal()

    def __init__(self, parent=None, persist=None):
        super().__init__(None, Qt.Window)
        self.persist = persist
        self.setWindowTitle("Live Optical Properties (μa, μs′) — " + APP_NAME)
        self.resize(850, 580)
        self.setMinimumSize(500, 360)
        self.setStyleSheet(f"background-color: {C['bg']}; color: {C['text']};")

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 12)
        v.setSpacing(8)

        # Header bar
        hdr = QtWidgets.QHBoxLayout()
        hdr.setContentsMargins(2, 0, 2, 0)
        hdr.setSpacing(12)

        lbl_title = QtWidgets.QLabel(
            f"<b style='color:{C['accent']}; font-size:11pt;'>Live Optical Properties</b>"
        )
        lbl_title.setMinimumWidth(180)
        hdr.addWidget(lbl_title)

        # Mini KPI indicators
        self.lbl_kpi_mua = QtWidgets.QLabel(f"<span style='color:{C['muted']};'>μa:</span> <b style='color:{C['mua']};'>—</b>")
        self.lbl_kpi_musp = QtWidgets.QLabel(f"<span style='color:{C['muted']};'>μs′:</span> <b style='color:{C['musp']};'>—</b>")
        self.lbl_kpi_dt = QtWidgets.QLabel(f"<span style='color:{C['muted']};'>Δ⟨t⟩:</span> <b style='color:{C['dt']};'>—</b>")
        self.lbl_kpi_frame = QtWidgets.QLabel(f"<span style='color:{C['muted']};'>Frame:</span> 0")

        for lbl in (self.lbl_kpi_mua, self.lbl_kpi_musp, self.lbl_kpi_dt, self.lbl_kpi_frame):
            lbl.setTextFormat(Qt.RichText)
            hdr.addWidget(lbl)

        hdr.addStretch(1)

        self.chk_ontop = QtWidgets.QCheckBox("Keep on top")
        self.chk_ontop.setToolTip("Keep this window floating above other windows")
        self.chk_ontop.toggled.connect(self._toggle_ontop)
        hdr.addWidget(self.chk_ontop)

        self.btn_dock = QtWidgets.QPushButton("⤓  Dock into Main Window")
        self.btn_dock.setObjectName("small")
        self.btn_dock.setToolTip("Dock the graphs back side-by-side with the histogram on the main screen")
        self.btn_dock.clicked.connect(self.dock_requested.emit)
        hdr.addWidget(self.btn_dock)

        v.addLayout(hdr)

        # Container for the plot widget
        self.plot_container = QtWidgets.QVBoxLayout()
        self.plot_container.setContentsMargins(0, 0, 0, 0)
        v.addLayout(self.plot_container, 1)

        self._plot_widget = None

    def set_plot_widget(self, w):
        if self._plot_widget and self._plot_widget != w:
            self.plot_container.removeWidget(self._plot_widget)
        self._plot_widget = w
        if w:
            w.setParent(self)
            self.plot_container.addWidget(w, 1)
            w.show()

    def update_kpis(self, lf, fr, n_frames):
        if lf:
            m = lf.get("mua", float("nan"))
            s = lf.get("musp", float("nan"))
            dt_ps = lf.get("dT_ps", 0.0)
            self.lbl_kpi_mua.setText(
                f"<span style='color:{C['muted']};'>μa:</span> <b style='color:{C['mua']};'>{m:.4f} cm⁻¹</b>"
                if np.isfinite(m) else f"<span style='color:{C['muted']};'>μa:</span> —"
            )
            self.lbl_kpi_musp.setText(
                f"<span style='color:{C['muted']};'>μs′:</span> <b style='color:{C['musp']};'>{s:.2f} cm⁻¹</b>"
                if np.isfinite(s) else f"<span style='color:{C['muted']};'>μs′:</span> —"
            )
            self.lbl_kpi_dt.setText(
                f"<span style='color:{C['muted']};'>Δ⟨t⟩:</span> <b style='color:{C['dt']};'>{dt_ps:.0f} ps</b>"
            )
        else:
            self.lbl_kpi_mua.setText(f"<span style='color:{C['muted']};'>μa:</span> —")
            self.lbl_kpi_musp.setText(f"<span style='color:{C['muted']};'>μs′:</span> —")
            self.lbl_kpi_dt.setText(f"<span style='color:{C['muted']};'>Δ⟨t⟩:</span> —")

        t_s = fr.get("t_s", 0.0) if fr else 0.0
        self.lbl_kpi_frame.setText(f"<span style='color:{C['muted']};'>Frame:</span> {n_frames} ({t_s:.1f} s)")

    def _toggle_ontop(self, checked):
        pos = self.pos()
        size = self.size()
        self.setWindowFlag(Qt.WindowStaysOnTopHint, checked)
        self.show()
        self.move(pos)
        self.resize(size)

    def closeEvent(self, event):
        event.ignore()
        self.hide()
        self.closed.emit()

    def force_close(self):
        self.closeEvent = lambda event: event.accept()
        self.close()


class LiveTab(QtWidgets.QWidget):
    status = QtCore.Signal(str, str)
    calibrated = QtCore.Signal(str)

    def __init__(self, persist):
        super().__init__()
        self.persist = persist
        self.session = None
        self.cfg = None
        self.recording = False
        self.reset_data()
        h = QtWidgets.QHBoxLayout(self)
        h.setContentsMargins(16, 14, 16, 16)
        h.setSpacing(16)

        # Calibration form & solver runner
        self.cal_form = CalForm()
        self.cal_log = new_log()
        self.runner = Runner(self.cal_log)
        self.runner.finished.connect(self.cal_done)
        self.cal_jobs = []

        # ---------- left sidebar: clear 1-2-3-4 workflow
        side = QtWidgets.QWidget()
        sv = QtWidgets.QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 6, 0)
        sv.setSpacing(12)
        self.w = {}

        # 1. Sensor & Acquisition
        c, lay = card("Sensor & Acquisition")

        # Sensor Auto-Detection & Connection Status Badge
        self.sensor_badge = QtWidgets.QFrame()
        self.sensor_badge.setObjectName("subcard")
        self.sensor_badge.setStyleSheet(
            f"QFrame#subcard {{ background: {C['panel']}; border: 1px solid {C['border']}; border-radius: 6px; }}"
        )
        b_lay = QtWidgets.QHBoxLayout(self.sensor_badge)
        b_lay.setContentsMargins(10, 8, 10, 8)
        b_lay.setSpacing(8)
        self.sensor_dot = QtWidgets.QLabel("◉")
        self.sensor_dot.setStyleSheet(f"color: {C['muted']}; font-size: 11pt;")
        self.sensor_info_lbl = QtWidgets.QLabel("Not Connected · Ready (127.0.0.1:39998)")
        self.sensor_info_lbl.setStyleSheet(f"color: {C['muted']}; font-weight: 500;")
        b_lay.addWidget(self.sensor_dot)
        b_lay.addWidget(self.sensor_info_lbl, 1)
        lay.addWidget(self.sensor_badge)

        self.tcp_box = QtWidgets.QWidget()
        tv = QtWidgets.QVBoxLayout(self.tcp_box)
        tv.setContentsMargins(0, 4, 0, 0)
        tv.setSpacing(6)

        # Primary useful controls (AMS EVM GUI Control)
        fi = form_layout()

        self.w["sensor_model"] = QtWidgets.QComboBox()
        self.w["sensor_model"].addItems([
            "Auto-detect (TMF882x Family)",
            "TMF8828 (up to 8×8, 64 zones)",
            "TMF8821 (up to 3×6 / 4×4, 18 zones)",
            "TMF8820 (3×3 only, 9 zones)",
        ])
        self.w["sensor_model"].setToolTip(
            "Target sensor hardware model or automatic identification.\n"
            "• Auto-detect: Identifies the sensor model from streaming channels and pixel grid on connect.\n"
            "• TMF8828: 8×8 (64 zones), 4×4 (16 zones), 3×6 (18 zones), 3×3 (9 zones)\n"
            "• TMF8821: 4×4, 3×6, 3×3\n"
            "• TMF8820: 3×3 only"
        )
        self.w["sensor_model"].currentIndexChanged.connect(self._sensor_model_changed)
        fi.addRow("Sensor Model", self.w["sensor_model"])

        self.w["pixels"] = QtWidgets.QComboBox()
        self.w["pixels"].addItems(["3 x 3", "4 x 4", "3 x 6", "8 x 8"])
        self.w["pixels"].setToolTip(
            "Sensor pixel layout: 3×3 (9 zones), 4×4 (16 zones), 3×6 (18 zones), 8×8 (64 zones). "
            "Sent to the sensor as (P0)–(P3) on Start and whenever you change it while connected. "
            "Configures the spatial channel buttons and detection grid."
        )
        fi.addRow("Pixels", self.w["pixels"])

        self.w["sensor_mode"] = QtWidgets.QComboBox()
        self.w["sensor_mode"].addItems(["Default", "High Power", "Low Power"])
        self.w["sensor_mode"].setToolTip("Sensor operational mode preset: Default (550k pulses), High Power (4000k pulses), Low Power (50k pulses).")
        fi.addRow("Mode", self.w["sensor_mode"])

        self.w["spad_map"] = QtWidgets.QComboBox()
        self.w["spad_map"].addItems(["33°x32°", "41°x52°", "28°x28°", "52°x52°"])
        self.w["spad_map"].setToolTip("Sensor SPAD field-of-view configuration.")
        fi.addRow("SPAD Map", self.w["spad_map"])

        self.w["iters"] = isspin(1, 65535, 550, "×10³ pulses") if "isspin" in globals() else ispin(1, 65535, 550, "×10³ pulses")
        self.w["iters"].setToolTip(
            "Laser pulses integrated per histogram, in thousands (e.g. 550 or 4000 = 4×10⁶ pulses). "
            "Must match the Direct Light (DL) recording used in calibration."
        )
        fi.addRow("Iterations", self.w["iters"])

        self.w["thresh"] = ispin(0, 255, 6)
        self.w["thresh"].setToolTip("Sensor object-detection threshold (default 6; sent as (T<val>)). Does not affect raw histograms.")
        fi.addRow("Threshold", self.w["thresh"])
        tv.addLayout(fi)

        lay.addWidget(self.tcp_box)

        # Hidden backwards-compatibility widgets (so persistence and test suites don't crash)
        self.w["src"] = QtWidgets.QComboBox()
        self.w["src"].addItem("TMF882x Sensor (EVM GUI TCP)")
        self.w["src"].addItem("Replay Saved Recording (.csv)")
        self.w["src"].setVisible(False)
        self.w["replay"] = PathEdit("file")
        self.w["replay"].setVisible(False)
        self.w["speed"] = dspin(0, 100, 1, 1, 1, "× real-time")
        self.w["speed"].setVisible(False)
        self.rep_box = QtWidgets.QWidget()
        self.rep_box.setVisible(False)

        # Spatial channels display drawer inside Step 1
        self.chan_box = Collapsible("Spatial Channels Displayed")
        self.chan = ChannelPicker()
        self.chan_box.form.addRow(self.chan)
        lay.addWidget(self.chan_box)
        self.chan.changed.connect(self._chan_changed)
        self.chan.layout_from_data.connect(self._layout_from_data)

        # Advanced Settings collapsible (less-used stuff: TCP/triggers + advanced amsgui settings)
        self.adv = Collapsible("Advanced Settings", expanded=False)
        self.w["host"] = QtWidgets.QLineEdit("127.0.0.1")
        self.w["port"] = ispin(1, 65535, 39998)
        self.w["trigger"] = QtWidgets.QLineEdit("(M0)")
        self.w["setup"] = QtWidgets.QLineEdit("(H3)")
        self.w["quiet"] = dspin(0.01, 2, 0.07, 2, 0.01, "s")
        self.adv.form.addRow("Host", self.w["host"])
        self.adv.form.addRow("Port", self.w["port"])
        self.adv.form.addRow("Trigger cmd", self.w["trigger"])
        self.adv.form.addRow("Init cmd", self.w["setup"])
        self.adv.form.addRow("Read timeout", self.w["quiet"])

        hc = QtWidgets.QHBoxLayout()
        self.cmd_text = QtWidgets.QLineEdit()
        self.cmd_text.setPlaceholderText("e.g. (T6)")
        self.cmd_btn = QtWidgets.QPushButton("Send")
        self.cmd_btn.setObjectName("small")
        self.cmd_btn.setEnabled(False)
        self.cmd_btn.clicked.connect(self.send_cmd)
        hc.addWidget(self.cmd_text, 1)
        hc.addWidget(self.cmd_btn)
        self.adv.form.addRow("Manual cmd", hc)

        self.w["bar_range"] = ispin(100, 10000, 5000, "mm")
        self.w["bar_range"].setToolTip("EVM GUI distance bar chart display range (default 5000 mm).")
        self.adv.form.addRow("Bar Chart Range", self.w["bar_range"])

        h_desc = QtWidgets.QHBoxLayout()
        self.w["descatter"] = QtWidgets.QCheckBox("Enable filter")
        self.w["descatter_coeff"] = ispin(0, 100, 5, "%")
        self.w["descatter_coeff"].setToolTip("De-scattering filter coefficient percentage (default 5%).")
        h_desc.addWidget(self.w["descatter"])
        h_desc.addWidget(QtWidgets.QLabel("Coeff:"))
        h_desc.addWidget(self.w["descatter_coeff"], 1)
        self.adv.form.addRow("De-scattering", h_desc)

        h_seq = QtWidgets.QHBoxLayout()
        self.w["seq_count"] = ispin(1, 1000, 10)
        self.seq_btn = QtWidgets.QPushButton("Sequence")
        self.seq_btn.setObjectName("small")
        self.seq_btn.setEnabled(False)
        self.seq_btn.setToolTip("Trigger a burst sequence of measurements over TCP.")
        self.seq_btn.clicked.connect(self.run_sequence)
        h_seq.addWidget(QtWidgets.QLabel("Count:"))
        h_seq.addWidget(self.w["seq_count"], 1)
        h_seq.addWidget(self.seq_btn)
        self.adv.form.addRow("Burst sequence", h_seq)

        h_scal = QtWidgets.QHBoxLayout()
        self.sensor_cal_btn = QtWidgets.QPushButton("Calibrate")
        self.sensor_cal_btn.setObjectName("small")
        self.sensor_cal_btn.setToolTip("Send sensor on-chip optical crosstalk calibration command.")
        self.sensor_cal_btn.clicked.connect(self.sensor_calibrate)
        self.sensor_clear_cal_btn = QtWidgets.QPushButton("Clear Cal")
        self.sensor_clear_cal_btn.setObjectName("small")
        self.sensor_clear_cal_btn.setToolTip("Clear sensor on-chip calibration.")
        self.sensor_clear_cal_btn.clicked.connect(self.sensor_clear_calibrate)
        h_scal.addWidget(self.sensor_cal_btn)
        h_scal.addWidget(self.sensor_clear_cal_btn)
        self.adv.form.addRow("Sensor on-chip cal", h_scal)

        lay.addWidget(self.adv)
        sv.addWidget(c)

        # Mode selector widget above Direct Light & Moments Fit
        mode_box = QtWidgets.QFrame()
        mode_box.setObjectName("subcard")
        mode_lay = QtWidgets.QVBoxLayout(mode_box)
        mode_lay.setContentsMargins(10, 8, 10, 8)
        mode_lay.setSpacing(6)

        lbl_mode = QtWidgets.QLabel("LIVE ACQUISITION MODE")
        lbl_mode.setObjectName("cardTitle")
        mode_lay.addWidget(lbl_mode)

        h_mode = QtWidgets.QHBoxLayout()
        h_mode.setSpacing(6)
        self.mode_btn_grp = QtWidgets.QButtonGroup(self)
        self.btn_mode_acquire = QtWidgets.QPushButton("Acquire Only")
        self.btn_mode_acquire.setCheckable(True)
        self.btn_mode_acquire.setChecked(True)
        self.btn_mode_acquire.setObjectName("modeBtn")
        self.btn_mode_acquire.setToolTip("Stream and record raw pToF histograms without running real-time fitting.")

        self.btn_mode_fit = QtWidgets.QPushButton("Analyse && Acquire")
        self.btn_mode_fit.setCheckable(True)
        self.btn_mode_fit.setChecked(False)
        self.btn_mode_fit.setObjectName("modeBtn")
        self.btn_mode_fit.setToolTip("Fit absorption (μa) and scattering (μs′) in real time from a calibrated sensor.")

        self.mode_btn_grp.addButton(self.btn_mode_acquire, 0)
        self.mode_btn_grp.addButton(self.btn_mode_fit, 1)
        self.btn_mode_acquire.clicked.connect(lambda: self._set_mode("acquire"))
        self.btn_mode_fit.clicked.connect(lambda: self._set_mode("fit"))

        h_mode.addWidget(self.btn_mode_acquire, 1)
        h_mode.addWidget(self.btn_mode_fit, 1)
        mode_lay.addLayout(h_mode)
        sv.addWidget(mode_box)

        # Direct Light (DL)
        c_dl, lay_dl = card("Direct Light (DL)")
        self.c_dl = c_dl
        self.w["dl"] = PathEdit("file", "CSV files (*.csv);;All files (*)", "Select Direct Light recording (.csv)…")
        lay_dl.addWidget(self.w["dl"])
        self.dl_info = muted("")
        self.dl_info.setVisible(False)
        lay_dl.addWidget(self.dl_info)
        sv.addWidget(c_dl)

        # 3. Data Recording
        c_rec, lay_rec = card("Data Recording")
        f_rec = form_layout()
        self.w["rec_type"] = QtWidgets.QComboBox()
        self.w["rec_type"].addItems([
            "Sample Measurement",
            "Reference Phantom (Calibration)",
            "Direct Light (DL)"
        ])
        f_rec.addRow("Recording type", self.w["rec_type"])
        lay_rec.addLayout(f_rec)

        # Duration checkbox + conditional input box
        self.w["dur_on"] = QtWidgets.QCheckBox("Auto-stop recording after set duration")
        lay_rec.addWidget(self.w["dur_on"])
        self.dur_box = QtWidgets.QWidget()
        hd = QtWidgets.QHBoxLayout(self.dur_box)
        hd.setContentsMargins(18, 0, 0, 0)
        hd.setSpacing(6)
        self.w["dur"] = ispin(1, 86400, 60)
        hd.addWidget(QtWidgets.QLabel("Duration:"))
        hd.addWidget(self.w["dur"], 1)
        hd.addWidget(QtWidgets.QLabel("seconds"))
        self.dur_box.setVisible(False)
        self.w["dur_on"].toggled.connect(self.dur_box.setVisible)
        lay_rec.addWidget(self.dur_box)

        self.rec_btn = QtWidgets.QPushButton("●  Start Recording…")
        self.rec_btn.setObjectName("record")
        self.rec_btn.setEnabled(False)
        self.rec_btn.clicked.connect(self.toggle_record)
        lay_rec.addWidget(self.rec_btn)

        # Reference Phantom Calibration Suite drawer (opens when Reference Phantom is selected)
        self.ref_panel = QtWidgets.QWidget()
        ref_v = QtWidgets.QVBoxLayout(self.ref_panel)
        ref_v.setContentsMargins(0, 10, 0, 0)
        ref_v.setSpacing(8)

        ref_header = QtWidgets.QLabel(
            f"<div style='padding:6px 10px; background:{C['card2']}; border-left:3px solid {C['accent']}; border-radius:4px;'>"
            f"<b>Reference Phantom Calibration</b></div>"
        )
        ref_header.setTextFormat(Qt.RichText)
        ref_v.addWidget(ref_header)

        # Target slot selector when recording
        target_box = QtWidgets.QWidget()
        tb_h = QtWidgets.QHBoxLayout(target_box)
        tb_h.setContentsMargins(0, 0, 0, 0)
        tb_h.setSpacing(10)
        self.rec_target_grp = QtWidgets.QButtonGroup(self)
        self.rec_target_primary = QtWidgets.QRadioButton("Primary Reference")
        self.rec_target_sec = QtWidgets.QRadioButton("Secondary Reference")
        self.rec_target_primary.setChecked(True)
        self.rec_target_sec.setEnabled(False)
        self.rec_target_grp.addButton(self.rec_target_primary, 0)
        self.rec_target_grp.addButton(self.rec_target_sec, 1)
        tb_h.addWidget(self.rec_target_primary)
        tb_h.addWidget(self.rec_target_sec)
        tb_h.addStretch(1)
        ref_v.addWidget(target_box)
        self.cal_form.refs.enable_sec.toggled.connect(self._on_enable_sec_toggled)

        ref_v.addWidget(self.cal_form)

        self.cal_run_btn = QtWidgets.QPushButton("Calibrate")
        self.cal_run_btn.setObjectName("primary")
        self.cal_run_btn.setToolTip("Run calibration on the reference recordings and save standard calibration (.json)")
        self.cal_run_btn.clicked.connect(self.run_calibration)
        ref_v.addWidget(self.cal_run_btn)

        self.cal_no_settings = QtWidgets.QCheckBox("Save without settings")
        self.cal_no_settings.setToolTip("Save calibration factors without embedding custom probe and solver settings into the file.")
        ref_v.addWidget(self.cal_no_settings)

        cal_status_row = QtWidgets.QHBoxLayout()
        self.cal_state_pill = pill("", C["muted"])
        self.cal_state_pill.setVisible(False)
        self.cal_result = QtWidgets.QLabel("")
        self.cal_result.setWordWrap(True)
        self.cal_result.setTextFormat(Qt.RichText)
        cal_status_row.addWidget(self.cal_state_pill)
        cal_status_row.addWidget(self.cal_result, 1)
        ref_v.addLayout(cal_status_row)

        self.ref_panel.setVisible(False)
        lay_rec.addWidget(self.ref_panel)
        self.w["rec_type"].currentIndexChanged.connect(self._rec_type_changed)
        sv.addWidget(c_rec)

        # 4. Real-Time Moments Fit
        c_fit, lay_fit = card("Real-Time Moments Fit")
        self.c_fit = c_fit
        f_cal = form_layout()
        self.w["cal"] = PathEdit("file", "Calibration (*.json);;All files (*)", "Select calibration (.json)…")
        f_cal.addRow("Calibration", self.w["cal"])
        lay_fit.addLayout(f_cal)

        self.w["cal_sec_on"] = QtWidgets.QCheckBox("Secondary calibration file (.json)")
        self.cal_sec_box = QtWidgets.QWidget()
        cs_lay = QtWidgets.QVBoxLayout(self.cal_sec_box)
        cs_lay.setContentsMargins(18, 0, 0, 0)
        self.w["cal_sec"] = PathEdit("file", "Calibration (*.json);;All files (*)", "Select secondary calibration (.json)… (leave blank if none)")
        cs_lay.addWidget(self.w["cal_sec"])
        self.cal_sec_box.setVisible(False)
        self.w["cal_sec_on"].toggled.connect(self.cal_sec_box.setVisible)
        lay_fit.addWidget(self.w["cal_sec_on"])
        lay_fit.addWidget(self.cal_sec_box)

        self.sc_cal, sclay = subcard()
        self.cal_info = QtWidgets.QLabel("")
        self.cal_info.setWordWrap(True)
        self.cal_info.setTextFormat(Qt.RichText)
        self.cal_info.setObjectName("muted")
        sclay.addWidget(self.cal_info)
        lay_fit.addWidget(self.sc_cal)
        self.sc_cal.setVisible(False)



        self.w["fit_on"] = QtWidgets.QCheckBox("Enable real-time optical property fit (μa, μs′)")
        lay_fit.addWidget(self.w["fit_on"])

        f_fit = form_layout()
        self.w["n"] = dspin(1.0, 2.0, 1.41, 3, 0.01)
        self.w["n"].setToolTip("Refractive index n of the measured sample (populated from calibration).")
        f_fit.addRow("Sample index (n)", self.w["n"])
        self.w["n_out"] = dspin(1.0, 2.0, 1.0, 3, 0.01)
        self.w["n_out"].setToolTip("Refractive index n_ext (air = 1.000, water = 1.333).")
        f_fit.addRow("External index (n_ext)", self.w["n_out"])
        self.w["mode"] = QtWidgets.QComboBox()
        self.w["mode"].addItems(["μa and μs′ (free fit)", "μa only (fix μs′)", "μs′ only (fix μa)"])
        f_fit.addRow("Fitted parameters", self.w["mode"])
        self.w["fix_val"] = dspin(0.001, 50, 10, 3, 0.1, "cm⁻¹")
        self.fix_lbl = QtWidgets.QLabel("Fixed μs′")
        f_fit.addRow(self.fix_lbl, self.w["fix_val"])

        h_win = QtWidgets.QHBoxLayout()
        self.w["win_pre"] = ispin(0, 50, 3, "bins")
        self.w["win_post"] = ispin(5, 120, 25, "bins")
        self.w["win_pre"].setToolTip("Moments integration window: bins before reference peak (default 3 bins ≈ 600 ps)")
        self.w["win_post"].setToolTip("Moments integration window: bins after reference peak (default 25 bins ≈ 5 ns)")
        self.w["win_pre"].valueChanged.connect(self.params_changed)
        self.w["win_post"].valueChanged.connect(self.params_changed)
        h_win.addWidget(QtWidgets.QLabel("Pre:"))
        h_win.addWidget(self.w["win_pre"], 1)
        h_win.addWidget(QtWidgets.QLabel("Post:"))
        h_win.addWidget(self.w["win_post"], 1)
        f_fit.addRow("Fit window", h_win)

        lay_fit.addLayout(f_fit)
        sv.addWidget(c_fit)

        sv.addStretch(1)

        outer = QtWidgets.QWidget()
        ov = QtWidgets.QVBoxLayout(outer)
        ov.setContentsMargins(0, 0, 0, 0)
        ov.setSpacing(10)
        ov.addWidget(scroll(side, 440), 1)
        self.start_btn = QtWidgets.QPushButton("▶  Start Acquisition")
        self.start_btn.setObjectName("primary")
        self.stop_btn = QtWidgets.QPushButton("■  Stop")
        self.stop_btn.setObjectName("danger")
        self.stop_btn.setEnabled(False)
        self.start_btn.clicked.connect(self.start)
        self.stop_btn.clicked.connect(self.stop)
        hb = QtWidgets.QHBoxLayout()
        hb.addWidget(self.start_btn, 2)
        hb.addWidget(self.stop_btn, 1)
        ov.addLayout(hb)
        outer.setFixedWidth(440)
        h.addWidget(outer)

        # ---------- main area
        main = QtWidgets.QWidget()
        mv = QtWidgets.QVBoxLayout(main)
        mv.setContentsMargins(0, 0, 0, 0)
        mv.setSpacing(12)
        self.kpis = QtWidgets.QWidget()
        kp = QtWidgets.QHBoxLayout(self.kpis)
        kp.setContentsMargins(0, 0, 0, 0)
        kp.setSpacing(12)
        self.k_mua = Kpi("ABSORPTION  μa", C["mua"], "cm⁻¹")
        self.k_musp = Kpi("REDUCED SCATTERING  μs′", C["musp"], "cm⁻¹")
        self.k_dt = Kpi("MEAN TIME SHIFT  Δ⟨t⟩", C["dt"], "ps vs. HOCAL")
        self.k_I = Kpi("PHOTON COUNTS  (I)", C["inten"], "counts")
        self.k_rate = Kpi("ACQUISITION RATE", C["muted"], "Hz")
        self.k_n = self.k_rate  # alias for backwards compatibility
        self.k_rate.setToolTip(
            "Live acquisition frame frequency & throughput:\n"
            "• Frequency (Hz): Rate of complete, independent histogram frames\n"
            "• Curves per sec: Total histogram curves arriving across all active channels (channels × Hz)\n"
            "• Note on 4×4 mode: The sensor captures 8 zones per packet at ~4.1 Hz; "
            "two packets complete one full 16-zone frame (~2.17 Hz full-frame rate)."
        )
        self._frame_times = []
        self._last_fps = 0.0
        for k in (self.k_mua, self.k_musp, self.k_dt, self.k_I, self.k_rate):
            k.setMinimumWidth(210)
            kp.addWidget(k, 1)

        self.scroll_kpis = QtWidgets.QScrollArea()
        self.scroll_kpis.setWidget(self.kpis)
        self.scroll_kpis.setWidgetResizable(True)
        self.scroll_kpis.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_kpis.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_kpis.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.scroll_kpis.setFixedHeight(106)
        self.scroll_kpis.setStyleSheet(f"""
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            QScrollBar:horizontal {{
                height: 10px;
                background: {C['panel']};
                border: 1px solid {C['border']};
                border-radius: 5px;
                margin: 1px 0px;
            }}
            QScrollBar::handle:horizontal {{
                background: #3b4254;
                min-width: 48px;
                border-radius: 4px;
            }}
            QScrollBar::handle:horizontal:hover {{
                background: {C['accent']};
            }}
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
                width: 0px;
                height: 0px;
            }}
        """)

        # Mouse wheel horizontal scrolling over parameter boxes
        def _on_kpi_wheel(ev):
            delta = ev.angleDelta().y() or ev.angleDelta().x()
            sb = self.scroll_kpis.horizontalScrollBar()
            sb.setValue(sb.value() - delta)
            ev.accept()
        self.scroll_kpis.wheelEvent = _on_kpi_wheel
        self.kpis.wheelEvent = _on_kpi_wheel

        # Left / Right chevron scroll buttons
        kpi_row = QtWidgets.QHBoxLayout()
        kpi_row.setContentsMargins(0, 0, 0, 0)
        kpi_row.setSpacing(6)

        btn_style = f"""
            QToolButton {{
                background: {C['panel']};
                border: 1px solid {C['border']};
                border-radius: 6px;
                color: {C['text']};
                font-size: 12pt;
                font-weight: bold;
            }}
            QToolButton:hover {{
                background: {C['card']};
                border-color: {C['accent']};
                color: {C['accent']};
            }}
            QToolButton:disabled {{
                color: #3b4254;
                border-color: transparent;
                background: transparent;
            }}
        """

        self.btn_kpi_prev = QtWidgets.QToolButton()
        self.btn_kpi_prev.setText("◀")
        self.btn_kpi_prev.setToolTip("Scroll parameter boxes left")
        self.btn_kpi_prev.setFixedSize(26, 84)
        self.btn_kpi_prev.setCursor(Qt.PointingHandCursor)
        self.btn_kpi_prev.setStyleSheet(btn_style)
        self.btn_kpi_prev.clicked.connect(lambda: self.scroll_kpis.horizontalScrollBar().setValue(
            self.scroll_kpis.horizontalScrollBar().value() - 220
        ))

        self.btn_kpi_next = QtWidgets.QToolButton()
        self.btn_kpi_next.setText("▶")
        self.btn_kpi_next.setToolTip("Scroll parameter boxes right")
        self.btn_kpi_next.setFixedSize(26, 84)
        self.btn_kpi_next.setCursor(Qt.PointingHandCursor)
        self.btn_kpi_next.setStyleSheet(btn_style)
        self.btn_kpi_next.clicked.connect(lambda: self.scroll_kpis.horizontalScrollBar().setValue(
            self.scroll_kpis.horizontalScrollBar().value() + 220
        ))

        def _update_kpi_scroll_btns(*_):
            sb = self.scroll_kpis.horizontalScrollBar()
            self.btn_kpi_prev.setEnabled(sb.value() > 0)
            self.btn_kpi_next.setEnabled(sb.value() < sb.maximum())
        self.scroll_kpis.horizontalScrollBar().valueChanged.connect(_update_kpi_scroll_btns)
        self.scroll_kpis.horizontalScrollBar().rangeChanged.connect(_update_kpi_scroll_btns)
        QtCore.QTimer.singleShot(100, _update_kpi_scroll_btns)

        kpi_row.addWidget(self.btn_kpi_prev)
        kpi_row.addWidget(self.scroll_kpis, 1)
        kpi_row.addWidget(self.btn_kpi_next)
        mv.addLayout(kpi_row)

        self.split = QtWidgets.QSplitter(Qt.Horizontal)
        gh = pg.GraphicsLayoutWidget()
        gh.setBackground(C["card"])
        self.p_h = style_plot(gh.addPlot(row=0, col=0, axisItems={"left": CountAxisItem(orientation="left")}), "photon counts", "time (ns)")
        self.h_ctl = HistControls()
        self.h_ctl.changed.connect(lambda: (self.h_ctl.apply_axes(self.p_h),
                                           self.show_frame(self.last_frame, count=False)
                                           if getattr(self, "last_frame", None) else None))
        self.p_h.setTitle(f"<span style='color:{C['muted']}'>pToF histogram</span>", size="9pt")
        self.p_h.addLegend(offset=(-10, 10))
        self.c_h = self.p_h.plot(pen=pg.mkPen(C["text"], width=1.5), fillLevel=0, symbol="o", symbolSize=4,
                                 symbolPen=None, symbolBrush=C["text"],
                                 brush=C["accent"] + "30")
        self.c_raw = self.p_h.plot(pen=pg.mkPen(C["muted"], width=1.3), symbol="o", symbolSize=3,
                                   symbolPen=None, symbolBrush=C["muted"])
        self.c_bg = self.p_h.plot(pen=pg.mkPen(C["warn"], width=1.5, style=Qt.DashLine), symbol="s", symbolSize=3,
                                  symbolPen=None, symbolBrush=C["warn"])
        self.c_clean = self.p_h.plot(pen=pg.mkPen(C["accent"], width=2.0), fillLevel=0, symbol="o", symbolSize=4,
                                     symbolPen=None, symbolBrush=C["accent"],
                                     brush=C["accent"] + "28")
        self.c_raw.setVisible(False)
        self.c_bg.setVisible(False)
        self.c_clean.setVisible(False)
        self.c_ref = self.p_h.plot(pen=pg.mkPen(C["muted"], width=1.2, style=Qt.DashLine),
                                   symbol="o", symbolSize=3, symbolPen=None, symbolBrush=C["muted"])
        self.c_ref.setVisible(False)
        self.win = pg.LinearRegionItem(movable=True, brush=C["warn"] + "18", pen=pg.mkPen(C["warn"] + "80"))
        self.win.sigRegionChangeFinished.connect(self._on_win_dragged)
        self.p_h.addItem(self.win)
        self.win.hide()
        self.h_ctl.apply_axes(self.p_h)
        self.btn_pop_opt = QtWidgets.QPushButton("⧉  Open μa & μs′ Window")
        self.btn_pop_opt.setObjectName("small")
        self.btn_pop_opt.setToolTip("Open separate window showing live absorption (μa) and scattering (μs′) graphs")
        self.btn_pop_opt.clicked.connect(self.show_optical_window)
        self.btn_pop_opt.setVisible(True)

        self.hist_box = with_bar(plot_bar([self.h_ctl, self.btn_pop_opt], PlotTools(gh, [("signal", self.p_h)], "live_signal")), gh)
        self.split.addWidget(self.hist_box)

        self.opt_window = LiveOpticalWindow(None, self.persist)
        self.opt_window.dock_requested.connect(self.dock_optical_window)
        self.opt_window.closed.connect(self._on_optical_window_closed)
        self.opt_popped_out = True

        self.opt = pg.GraphicsLayoutWidget()
        self.opt.setBackground(C["card"])
        self.p_mua = style_plot(self.opt.addPlot(row=0, col=0), "μa (cm⁻¹)")
        self.p_musp = style_plot(self.opt.addPlot(row=1, col=0), "μs′ (cm⁻¹)", "time (s)")
        self.p_musp.setXLink(self.p_mua)
        self.p_mua.getAxis("bottom").setStyle(showValues=False)
        sym = dict(symbol="o", symbolSize=4, symbolPen=None)
        self.c_mua = self.p_mua.plot(pen=pg.mkPen(C["mua"], width=1.6), connect="finite", symbolBrush=C["mua"], **sym)
        self.c_musp = self.p_musp.plot(pen=pg.mkPen(C["musp"], width=1.6), connect="finite", symbolBrush=C["musp"], **sym)

        self.btn_undock_opt = QtWidgets.QPushButton("⧉  Pop out Window")
        self.btn_undock_opt.setObjectName("small")
        self.btn_undock_opt.setToolTip("Pop out μa and μs′ graphs into a separate window")
        self.btn_undock_opt.clicked.connect(self.undock_optical_window)
        self.btn_undock_opt.setVisible(False)

        self.opt_box = with_bar(plot_bar([self.btn_undock_opt],
                                         PlotTools(self.opt, [("μa", self.p_mua), ("μs′", self.p_musp)],
                                                   "live_optical_properties")), self.opt)
        self.opt_window.set_plot_widget(self.opt_box)
        mv.addWidget(self.split, 1)

        footer = QtWidgets.QHBoxLayout()
        self.msg = QtWidgets.QLabel("")
        self.msg.setObjectName("muted")
        footer.addWidget(self.msg, 1)
        mv.addLayout(footer)
        h.addWidget(main, 1)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.setInterval(60)
        persist.load("live", self.w)
        if self.persist.s.value("live/rec_type") == "Reference Phantom":
            self.w["rec_type"].setCurrentIndex(1)
        persist.load("cal2", self.cal_form.w)
        try:
            self.cal_form.refs.set_rows(json.loads(self.persist.s.value("cal2/references", "[]")))
        except Exception:
            pass
        if self.persist.s.value("cal2/diff_cal_dl") == "True":
            self.cal_form.diff_cal_dl.setChecked(True)
        if self.persist.s.value("cal2/enable_sec") == "True":
            self.cal_form.refs.enable_sec.setChecked(True)
        if self.persist.s.value("cal2/no_settings") == "True" and hasattr(self, "cal_no_settings"):
            self.cal_no_settings.setChecked(True)
        self.cal_form.changed.connect(self.save_state)
        self.dur_box.setVisible(self.w["dur_on"].isChecked())
        self.ref_panel.setVisible(self.w["rec_type"].currentIndex() == 1)
        self._src_changed()
        if self.w["dl"].text():
            self._on_dl_changed(self.w["dl"].text())
        if self.w["cal"].text():
            self.load_cal(self.w["cal"].text())
        if self.cfg is None:
            self.w["fit_on"].setChecked(False)
        self.w["dl"].changed.connect(self._on_dl_changed)
        self.w["cal"].changed.connect(self.load_cal)
        self.w["cal_sec"].changed.connect(lambda: self.load_cal(self.w["cal"].text()))
        self.w["cal_sec_on"].toggled.connect(lambda: self.load_cal(self.w["cal"].text()))
        self.w["fit_on"].toggled.connect(self.fit_toggled)
        self.w["n"].valueChanged.connect(lambda v: self.w["n_out"].setMaximum(v))
        self.w["n_out"].setMaximum(self.w["n"].value())
        for k in ("n", "n_out", "fix_val"):
            self.w[k].valueChanged.connect(self.params_changed)
        self.w["mode"].currentIndexChanged.connect(self._mode_changed)
        self.w["iters"].editingFinished.connect(self.iters_changed)
        self.w["thresh"].editingFinished.connect(
            lambda: self.session and self.w["src"].currentIndex() == 0
            and self.session.send_command(f"(T{self.w['thresh'].value()})")
        )
        self.w["sensor_mode"].currentTextChanged.connect(self._sensor_mode_changed)
        self.w["pixels"].currentTextChanged.connect(self._pixels_changed)
        self._pixels_changed()
        # EVM GUI settings with no TCP command: say so instead of silently doing nothing
        for k, lab in (("spad_map", "SPAD map"), ("bar_range", "Bar chart range"),
                       ("descatter", "De-scattering filter"), ("descatter_coeff", "De-scattering coefficient")):
            wdg = self.w[k]
            wdg.setToolTip(wdg.toolTip() + ("\n" if wdg.toolTip() else "")
                           + "Set this in the AMS EVM GUI: there is no TCP command for it, so momenTOF does not send it.")
            sig = (wdg.currentTextChanged if isinstance(wdg, QtWidgets.QComboBox) else
                   wdg.toggled if isinstance(wdg, QtWidgets.QAbstractButton) else wdg.valueChanged)
            sig.connect(lambda *_, lab=lab: self.msg.setText(
                f"{lab}: set the same value in the AMS EVM GUI (it is not sent over TCP)."))
        self._apply_fit_view()

    # ---- helpers
    def reset_data(self):
        self.T, self.MUA, self.MUSP = [], [], []
        self.n_frames = 0
        self.rec_started = None
        self._frame_times = []
        self._last_fps = 0.0
        if hasattr(self, "c_mua") and hasattr(self, "c_musp"):
            self.c_mua.setData([], [])
            self.c_musp.setData([], [])
        if hasattr(self, "opt_window") and self.opt_window:
            self.opt_window.update_kpis(None, None, 0)
        if hasattr(self, "k_rate"):
            self.k_rate.set("0.00 Hz", "0 curves/s · 0 frames")

    def show_optical_window(self):
        if not self.opt_popped_out:
            self.undock_optical_window()
        self.opt_window.show()
        self.opt_window.raise_()
        self.opt_window.activateWindow()

    def _on_optical_window_closed(self):
        if hasattr(self, "btn_pop_opt"):
            self.btn_pop_opt.setText("⧉  Open μa & μs′ Window")

    def dock_optical_window(self):
        self.opt_popped_out = False
        self.opt_window.hide()
        self.opt_box.setParent(self.split)
        self.split.addWidget(self.opt_box)
        self.opt_box.show()
        self.btn_undock_opt.setVisible(True)
        self.split.setSizes([500, 620])
        self.btn_pop_opt.setText("⧉  Pop out μa & μs′ Window")

    def undock_optical_window(self):
        self.opt_popped_out = True
        self.opt_box.setParent(None)
        self.opt_window.set_plot_widget(self.opt_box)
        self.btn_undock_opt.setVisible(False)
        self.btn_pop_opt.setText("⧉  Open μa & μs′ Window")
        self.opt_window.show()
        self.opt_window.raise_()
        self.opt_window.activateWindow()

    def _on_enable_sec_toggled(self, checked):
        self.rec_target_sec.setEnabled(checked)
        if not checked:
            self.rec_target_primary.setChecked(True)

    def _rec_type_changed(self, i=None):
        i = self.w["rec_type"].currentIndex() if i is None else i
        self.ref_panel.setVisible(i == 1)
        self._apply_fit_view()

    def _sensor_model_changed(self, idx):
        # 0: Auto, 1: TMF8828, 2: TMF8821, 3: TMF8820
        allowed = {
            0: ["3 x 3", "4 x 4", "3 x 6", "8 x 8"],
            1: ["3 x 3", "4 x 4", "3 x 6", "8 x 8"],
            2: ["3 x 3", "4 x 4", "3 x 6"],
            3: ["3 x 3"],
        }.get(idx, ["3 x 3", "4 x 4", "3 x 6", "8 x 8"])

        cur = self.w["pixels"].currentText()
        self.w["pixels"].blockSignals(True)
        self.w["pixels"].clear()
        self.w["pixels"].addItems(allowed)
        if cur in allowed:
            self.w["pixels"].setCurrentText(cur)
        else:
            self.w["pixels"].setCurrentIndex(0)
            self._pixels_changed()
        self.w["pixels"].blockSignals(False)
        self._update_sensor_badge()

    def _update_sensor_badge(self, frame=None, itt=None):
        if not getattr(self, "session", None):
            self.sensor_dot.setText("◉")
            self.sensor_dot.setStyleSheet(f"color: {C['muted']}; font-size: 11pt;")
            model_text = self.w["sensor_model"].currentText().split(" ")[0]
            if model_text.startswith("Auto"):
                self.sensor_info_lbl.setText("Not Connected · Ready (127.0.0.1:39998)")
            else:
                self.sensor_info_lbl.setText(f"Not Connected · Target: {model_text} (port 39998)")
            self.sensor_info_lbl.setStyleSheet(f"color: {C['muted']}; font-weight: 500;")
            return

        pix = self.w["pixels"].currentText()
        n_ch = self.PIXEL_ZONES.get(pix, len(self.chan.buttons) if hasattr(self.chan, "buttons") else 16)
        if frame and "channels" in frame:
            n_ch = len(frame["channels"])

        override = self.w["sensor_model"].currentIndex()
        if override == 1:
            detected = f"TMF8828 Confirmed · {pix.replace(' x ', '×')} ({n_ch} zones)"
        elif override == 2:
            detected = f"TMF8821 Confirmed · {pix.replace(' x ', '×')} ({n_ch} zones)"
        elif override == 3:
            detected = f"TMF8820 Confirmed · {pix.replace(' x ', '×')} ({n_ch} zones)"
        else:
            if pix == "8 x 8" or n_ch > 18:
                detected = f"TMF8828 Detected · 8×8 ({n_ch} zones)"
            elif pix in ("4 x 4", "3 x 6") or n_ch > 9:
                detected = f"TMF8828 Detected · {pix.replace(' x ', '×')} ({n_ch} zones)"
            else:
                detected = f"TMF882x Detected · {pix.replace(' x ', '×')} ({n_ch} zones)"

        self.sensor_dot.setText("●")
        self.sensor_dot.setStyleSheet(f"color: {C['ok']}; font-size: 11pt;")
        self.sensor_info_lbl.setText(detected)
        self.sensor_info_lbl.setStyleSheet(f"color: {C['ok']}; font-weight: 600;")

    def _src_changed(self):
        pass

    def _set_mode(self, mode):
        is_fit = (mode == "fit")
        self.w["fit_on"].setChecked(is_fit)
        self._apply_fit_view()

    def _apply_fit_view(self):
        on = self.w["fit_on"].isChecked()
        if not on:
            self.last_fit = None
        self.btn_pop_opt.setVisible(True)
        if self.opt_popped_out:
            if on:
                self.show_optical_window()
        else:
            self.opt_box.setVisible(on)
        if hasattr(self, "c_dl"):
            self.c_dl.setVisible(True)
        if hasattr(self, "c_fit"):
            self.c_fit.setVisible(True)
        if hasattr(self, "btn_mode_fit"):
            self.btn_mode_fit.blockSignals(True)
            self.btn_mode_acquire.blockSignals(True)
            self.btn_mode_fit.setChecked(on)
            self.btn_mode_acquire.setChecked(not on)
            self.btn_mode_fit.blockSignals(False)
            self.btn_mode_acquire.blockSignals(False)
        for k in (self.k_mua, self.k_musp, self.k_dt):
            k.setEnabled(on)
            if not on:
                k.set("—", "")
        mode = self.w["mode"].currentIndex()
        self.fix_lbl.setText("Fixed μs′" if mode == 1 else "Fixed μa")
        for wdg in (self.fix_lbl, getattr(self.w["fix_val"], "unit_container", self.w["fix_val"])):
            wdg.setVisible(mode != 0)

    def _on_dl_changed(self, p):
        if p and os.path.isfile(p):
            try:
                _, n = V.read_hlong(p)
                if n:
                    pulses = int(n)
                    self.w["iters"].setValue(pulses // 1000)
                    self.iters_changed()
                    self.dl_info.setText(
                        f"<span style='color:{C['ok']}'>✓ {pulses:,} pulses ({pulses // 1000}k) · matched to sensor iterations</span>"
                    )
                    self.dl_info.setVisible(True)
                else:
                    self.dl_info.setText(f"<span style='color:{C['ok']}'>✓ DL loaded: {os.path.basename(p)}</span>")
                    self.dl_info.setVisible(True)
            except Exception as e:
                self.dl_info.setText(f"<span style='color:{C['bad']}'>Could not read DL: {e}</span>")
                self.dl_info.setVisible(True)
        else:
            self.dl_info.setText("")
            self.dl_info.setVisible(False)
        if getattr(self, "session", None):
            self.session.set_dl(p if (p and os.path.isfile(p)) else None)
        self.params_changed()


    def load_cal(self, path):
        self.cfg = None
        if not path:
            self.cal_info.setText("")
            self.sc_cal.setVisible(False)
        else:
            try:
                self.cfg = L.load_calibration(path)
                summary_text = cal_summary(self.cfg)
                if self.w.get("cal_sec_on") and self.w["cal_sec_on"].isChecked() and self.w["cal_sec"].text():
                    try:
                        cfg2 = L.load_calibration(self.w["cal_sec"].text())
                        summary_text += f"<br><span style='color:{C['accent']}'><b>Secondary cal:</b> {cal_summary(cfg2)}</span>"
                    except Exception:
                        pass
                self.cal_info.setText(summary_text)
                self.sc_cal.setVisible(True)
                if not self.cfg.get("no_settings"):
                    self.w["n"].setValue(self.cfg["settings"]["N_TISSUE"])
                    self.w["n_out"].setValue(self.cfg["settings"].get("N_OUT", 1.0))
                    if "WIN_PRE" in self.cfg["settings"] and "win_pre" in self.w:
                        self.w["win_pre"].setValue(int(self.cfg["settings"]["WIN_PRE"]))
                    if "WIN_POST" in self.cfg["settings"] and "win_post" in self.w:
                        self.w["win_post"].setValue(int(self.cfg["settings"]["WIN_POST"]))
                    if self.cfg.get("dl_pulses") and not self.w["dl"].text():
                        self.w["iters"].setValue(int(self.cfg["dl_pulses"]) // 1000)
            except Exception as e:
                self.cal_info.setText(f"<span style='color:{C['bad']}'>Could not read calibration: {e}</span>")
                self.sc_cal.setVisible(True)
        zones = (self.cfg or {}).get("settings", {}).get("ZONES") if self.cfg else None
        try:
            self.chan.set_fit_zones([int(str(z)[6:]) for z in zones] if zones else None)
        except (ValueError, TypeError):
            self.chan.set_fit_zones(None)
        if self.session and self.w["fit_on"].isChecked():
            self.session.set_fitter(self.make_fitter() if self.cfg else None)

    def _on_win_dragged(self):
        if not getattr(self, "last_fit", None):
            return
        wb = self.win.getRegion()
        ctl = self.h_ctl
        scale = 1.0 if ctl.unit() == "bins" else 1000.0 / V.DT_PS
        b0, b1 = int(round(wb[0] * scale)), int(round(wb[1] * scale))
        ref = self.last_fit.get("ref")
        if ref is not None:
            k = int(np.argmax(ref))
            win_shift = getattr(V, "WIN_SHIFT", 0)
            pre = max(0, k + win_shift - b0)
            post = max(5, b1 - (k + win_shift))
            self.w["win_pre"].blockSignals(True)
            self.w["win_post"].blockSignals(True)
            self.w["win_pre"].setValue(pre)
            self.w["win_post"].setValue(post)
            self.w["win_pre"].blockSignals(False)
            self.w["win_post"].blockSignals(False)
            self.params_changed()

    def use_calibration(self, path):
        self.w["cal"].setText(path)
        self.load_cal(path)
        if self.cfg is not None and not self.w["fit_on"].isChecked():
            self.w["fit_on"].setChecked(True)

    def make_fitter(self):
        if not self.cfg:
            return None
        mode = self.w["mode"].currentIndex()
        v = self.w["fix_val"].value()
        frm = "both"
        cfg = dict(self.cfg)
        cfg["settings"] = dict(self.cfg.get("settings", {}))
        if "win_pre" in self.w and "win_post" in self.w:
            cfg["settings"]["WIN_PRE"] = int(self.w["win_pre"].value())
            cfg["settings"]["WIN_POST"] = int(self.w["win_post"].value())
        dl_path = self.w["dl"].text()
        if dl_path and os.path.isfile(dl_path):
            try:
                cfg = L.with_session_dl(cfg, dl_path)
            except Exception as e:
                self.msg.setText(f"Session DL not applied: {e}")
        return L.LiveFitter(cfg, 1, v if mode == 1 else None, self.w["n"].value(),
                            mua_fix=v if mode == 2 else None, fix_from=frm,
                            n_outside=min(self.w["n_out"].value(), self.w["n"].value()))

    def fit_toggled(self, on):
        self._apply_fit_view()
        if self.session:
            self.T, self.MUA, self.MUSP = [], [], []
            self.redraw()
            self.session.set_fitter(self.make_fitter() if (on and self.cfg) else None)
        self.save_state()

    def _mode_changed(self, i):
        self.w["fix_val"].blockSignals(True)
        self.w["fix_val"].setValue(10.0 if i == 1 else 0.1)
        self.w["fix_val"].blockSignals(False)
        self._apply_fit_view()
        self.params_changed()

    def params_changed(self, *_):
        if self.session and self.w["fit_on"].isChecked() and self.cfg is not None:
            self.session.set_fitter(self.make_fitter())
            self.msg.setText("Fit parameters updated — real-time fitting restarted with new values.")

    def iters_changed(self):
        if self.session and self.w["src"].currentIndex() == 0:
            self.session.send_command(f"(I{self.w['iters'].value()})")
            self.msg.setText(f"Sensor integration set to {self.w['iters'].value()}×10³ pulses.")

    # ---- run control
    def start(self):
        if self.session is not None:
            return
        self.persist.save("live", self.w)
        try:
            setup = [self._pixels_cmd(), f"(I{self.w['iters'].value()})",
                     f"(T{self.w['thresh'].value()})"] + self.w["setup"].text().split()
            src = L.EvmTcpSource(self.w["host"].text(), self.w["port"].value(), self.w["trigger"].text().strip(),
                                 tuple(setup), self.w["quiet"].value())
            fitter = self.make_fitter() if (self.w["fit_on"].isChecked() and self.cfg) else None
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, APP_NAME, f"{type(e).__name__}: {e}")
            return
        self.reset_data()
        self.chan.reset_data()
        self._itt_pix = None
        self.last_frame = None
        self.last_fit = None
        for p in (self.p_mua, self.p_musp, self.p_h):
            for it in list(p.items):
                if isinstance(it, pg.InfiniteLine):
                    p.removeItem(it)
        self.redraw()
        if self.w["fit_on"].isChecked() and self.opt_popped_out:
            self.show_optical_window()
        dl_p = self.w["dl"].text().strip()
        self.session = L.LiveSession(src, fitter, dl_path=dl_p if (dl_p and os.path.isfile(dl_p)) else None)
        self.session.start()
        self.timer.start()
        self._update_sensor_badge()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.cmd_btn.setEnabled(True)
        self.rec_btn.setEnabled(True)
        self.seq_btn.setEnabled(True)
        self.sensor_cal_btn.setEnabled(True)
        self.sensor_clear_cal_btn.setEnabled(True)
        self.msg.setText("Connecting to source…")
        self.status.emit("Connecting…", C["warn"])

    def stop(self):
        if self.session:
            self.session.stop()
            self.stop_btn.setEnabled(False)

    def toggle_record(self):
        if not self.session:
            return
        if self.recording:
            self.session.stop_recording()
            self.rec_btn.setEnabled(False)
            return
        kind_i = self.w["rec_type"].currentIndex()
        folder = self.persist.s.value("live/rec_dir", "") or os.path.expanduser("~")
        last = self.persist.s.value(f"live/rec_name{kind_i}", "")

        target_slot = 0
        if kind_i == 1:
            target_slot = 1 if (self.cal_form.refs.enable_sec.isChecked() and self.rec_target_sec.isChecked()) else 0
            suggest = last or ("B3" if target_slot == 1 else "B2")
            title = "Save Secondary Reference Phantom Recording As" if target_slot == 1 else "Save Primary Reference Phantom Recording As"
        elif kind_i == 2:
            suggest = last or "DL"
            title = "Save Direct Light (DL) Recording As"
        else:
            suggest = last or "sample"
            title = "Save Sample Measurement As"

        raw, _ = QtWidgets.QFileDialog.getSaveFileName(self, title, os.path.join(folder, suggest + ".csv"),
                                                       "Recording (*.csv)")
        if not raw:
            return
        raw = os.path.normpath(raw if raw.lower().endswith(".csv") else raw + ".csv")
        res = raw[:-4] + "_live_results.csv"
        self.persist.s.setValue("live/rec_dir", os.path.dirname(raw))
        self.persist.s.setValue(f"live/rec_name{kind_i}", os.path.splitext(os.path.basename(raw))[0])
        self.persist.save("live", self.w)

        if kind_i == 1:
            kind = "ref2" if target_slot == 1 else "ref1"
            slot_data = self.cal_form.refs.slot(target_slot)
            try:
                mua_val = float(slot_data[1]) if slot_data[1] else 0.0
            except ValueError:
                mua_val = 0.0
            try:
                musp_val = float(slot_data[2]) if slot_data[2] else 0.0
            except ValueError:
                musp_val = 0.0
        elif kind_i == 2:
            kind = "dl"
            mua_val, musp_val = 0.0, 0.0
        else:
            kind = "measurement"
            mua_val, musp_val = 0.0, 0.0

        self.rec_meta = (kind, raw, mua_val, musp_val)
        self.session.start_recording(raw, res if kind == "measurement" else None)
        self.recording = True
        self.rec_btn.setText("■  Stop Recording")
        self.rec_btn.setProperty("rec", True)
        self._restyle(self.rec_btn)
        self.msg.setText("Recording armed — waiting for next frame…")

    def _restyle(self, w):
        w.style().unpolish(w)
        w.style().polish(w)

    def _conn_text(self):
        return "Sensor Connected" if self.w["src"].currentIndex() == 0 else "Replaying File"

    def _rec_done(self, paths):
        self.recording = False
        self.rec_btn.setText("●  Start Recording…")
        self.rec_btn.setProperty("rec", False)
        self._restyle(self.rec_btn)
        self.rec_btn.setEnabled(self.session is not None)
        self.rec_started = None
        self.msg.setText("Saved " + ", ".join(os.path.basename(p) for p in paths))
        self.status.emit(self._conn_text() if self.session else "Disconnected", C["ok"] if self.session else C["bad"])
        meta = getattr(self, "rec_meta", None)
        self.rec_meta = None
        if meta and paths and meta[1] in paths:
            kind, raw, mua, musp = meta
            if kind in ("ref1", "ref2", "reference") and (mua or musp):
                self._save_certified(raw, mua, musp)
            if kind != "measurement":
                self.take_recording(kind, raw, mua, musp)

    def _save_certified(self, raw, mua, musp):
        """Keep the certified values next to the recordings, in the same format as milanfit_expected.json."""
        path = os.path.join(os.path.dirname(raw), "certified_values.json")
        try:
            with open(path, encoding="utf-8") as fh:
                d = json.load(fh) if os.path.isfile(path) else {}
        except Exception:
            d = {}
        d[os.path.splitext(os.path.basename(raw))[0]] = {"mean_mua": mua, "mean_musp": musp}
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(d, fh, indent=1)

    def open_calibration_form(self):
        self.w["rec_type"].setCurrentIndex(1)



    def take_recording(self, kind, raw, mua, musp):
        w = self.cal_form.w
        if kind == "dl":
            self.w["dl"].setText(raw)
            what = "Direct Light (DL)"
        elif kind == "ref1":
            self.cal_form.refs.set_slot(0, raw, mua, musp)
            what = "Primary Reference Phantom"
        elif kind == "ref2":
            self.cal_form.refs.set_slot(1, raw, mua, musp)
            what = "Secondary Reference Phantom"
        elif kind == "reference":
            slot_idx = self.cal_form.refs.add_recording(raw, mua, musp)
            what = "Primary Reference Phantom" if slot_idx == 0 else "Secondary Reference Phantom"
        else:
            what = "Recording"
        self.save_state()
        if kind in ("ref1", "ref2", "reference"):
            self.w["rec_type"].setCurrentIndex(1)
        self.msg.setText(f"Saved {os.path.basename(raw)} — set as {what}.")

    def run_calibration(self):
        if self.runner.running():
            return
        import tempfile
        self.cal_jobs = []
        try:
            session_dl = self.w["dl"].text()
            n_refs = 2 if (self.cal_form.refs.enable_sec.isChecked() and self.cal_form.refs.slot(1)[0]) else 1
            for i in range(n_refs):
                work = tempfile.mkdtemp(prefix=f"ptof_cal{i}_")
                self.cal_jobs.append((work, self.cal_form.prepare(work, i, session_dl=session_dl)))
            if len(self.cal_jobs) == 2 and os.path.abspath(self.cal_form.refs.slot(0)[0]) == \
                    os.path.abspath(self.cal_form.refs.slot(1)[0]):
                raise ValueError("The secondary reference phantom must be a different recording from the primary reference.")
        except ValueError as e:
            for wk, _ in self.cal_jobs:
                shutil.rmtree(wk, ignore_errors=True)
            self.cal_jobs = []
            QtWidgets.QMessageBox.warning(self, APP_NAME, str(e))
            return

        dl_path = self.cal_form.get_dl_path(fallback_dl=self.w["dl"].text())
        start = os.path.dirname(dl_path) if dl_path else (self.persist.s.value("live/rec_dir", "") or os.path.expanduser("~"))
        start = os.path.dirname(start) if os.path.basename(start).lower() == "dl" else start
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Calibration File As",
            os.path.join(start, f"calibration_{dt.datetime.now():%Y%m%d_%H%M}.json"),
            "Calibration (*.json)"
        )
        if not p:
            for wk, _ in self.cal_jobs:
                shutil.rmtree(wk, ignore_errors=True)
            self.cal_jobs = []
            return
        self.cal_target = p if p.lower().endswith(".json") else p + ".json"
        self.save_state()
        self.cal_stage, self.cal_cfgs = 0, []
        self.cal_log.clear()
        self.cal_run_btn.setEnabled(False)
        self._start_cal_stage()

    def _start_cal_stage(self):
        work, args = self.cal_jobs[self.cal_stage]
        self.tmp_cal = os.path.join(work, "calibration.json")
        what = "primary reference phantom" if self.cal_stage == 0 else "secondary reference phantom"
        set_pill(self.cal_state_pill, "Calibrating…", C["warn"])
        self.cal_state_pill.setVisible(True)
        self.cal_result.setText(f"<span style='color:{C['warn']}'>Running calibration ({what})…</span>")
        if self.cal_stage:
            self.cal_log.appendPlainText("\n===== Secondary Reference Calibration =====\n")
        self.runner.start([CAL_CLI, "--save-cal", self.tmp_cal] + args)

    def _finish_cfg(self, cfg, which):
        tdir = os.path.dirname(os.path.abspath(self.cal_target))
        p, a, b = self.cal_form.refs.slot(which)
        cfg["references"] = [{"file": p, "file_rel": L.relative_to(p, tdir), "mua": float(a), "musp": float(b)}]
        irf = self.cal_form.w["irf"].text() or None
        cfg["irf_file"] = irf
        cfg["irf_file_rel"] = L.relative_to(irf, tdir) if irf else None
        cfg["dl_file_rel"] = L.relative_to(cfg["dl_file"], tdir) if cfg.get("dl_file") else None
        cfg["v12_args"] = [x for x in cfg.get("v12_args", [])]
        return cfg

    def cal_done(self, code):
        failed = code != 0 or not os.path.isfile(self.tmp_cal)
        if not failed:
            self.cal_cfgs.append(self._finish_cfg(L.load_calibration(self.tmp_cal), self.cal_stage))
            if self.cal_stage + 1 < len(self.cal_jobs):
                self.cal_stage += 1
                self._start_cal_stage()
                return
        self.cal_run_btn.setEnabled(True)
        try:
            if failed:
                what = "primary reference" if self.cal_stage == 0 else "secondary reference"
                set_pill(self.cal_state_pill, "Failed", C["bad"])
                self.cal_state_pill.setVisible(True)
                self.cal_result.setText(f"<span style='color:{C['bad']}'>Calibration failed ({what}).</span>")
                return
            cfg = self.cal_cfgs[0]
            if hasattr(self, "cal_no_settings") and self.cal_no_settings.isChecked():
                cfg["no_settings"] = True
            else:
                cfg["gui_inputs"] = self.cal_form.values()
            if len(self.cal_cfgs) > 1:
                cfg["second_reference"] = self.cal_cfgs[1]
            with open(self.cal_target, "w", encoding="utf-8") as fh:
                json.dump(cfg, fh, indent=1, ensure_ascii=False)
            refs = " + secondary " + ", ".join(self.cal_cfgs[1]["refs"]) if len(self.cal_cfgs) > 1 else ""
            set_pill(self.cal_state_pill, "Saved & Active", C["ok"])
            self.cal_state_pill.setVisible(True)
            self.cal_result.setText(
                f"<span style='color:{C['ok']}'>Saved <b>{os.path.basename(self.cal_target)}</b> "
                f"(ref: {', '.join(cfg['refs'])}{refs}) — active for live &amp; batch analysis.</span>"
            )
            self.use_calibration(self.cal_target)
            self.calibrated.emit(self.cal_target)
        finally:
            for wk, _ in self.cal_jobs:
                shutil.rmtree(wk, ignore_errors=True)
            self.cal_jobs = []

    def save_state(self):
        self.persist.save("live", self.w)
        if hasattr(self, "cal_form"):
            self.persist.save("cal2", self.cal_form.w)
            self.persist.s.setValue("cal2/references", json.dumps(self.cal_form.refs.rows()))
            self.persist.s.setValue("cal2/diff_cal_dl", str(self.cal_form.diff_cal_dl.isChecked()))
            self.persist.s.setValue("cal2/enable_sec", str(self.cal_form.refs.enable_sec.isChecked()))
            if hasattr(self, "cal_no_settings"):
                self.persist.s.setValue("cal2/no_settings", str(self.cal_no_settings.isChecked()))

    def send_cmd(self):
        if self.session and self.cmd_text.text().strip():
            self.session.send_command(self.cmd_text.text().strip())
            self.msg.setText(f"Sent command: {self.cmd_text.text().strip()}")

    T_last = None

    # spad_map_id (the #ITT 'SPAD Map ID' column) -> field of view, TMF882x datasheet Fig. 30/31
    SPAD_MAP_FOV = {"1": "33°x32°", "2": "33°x47°", "3": "33°x47°", "4": "33°x47°", "5": "33°x47°",
                    "6": "41°x52°", "7": "41°x52°", "10": "33°x60°", "11": "33°x32°", "12": "33°x32°",
                    "13": "33°x42°"}

    PIXEL_ZONES = {
        "3 x 3": 9,
        "4 x 4": 16,
        "3 x 6": 18,
        "8 x 8": 64,
    }

    def _pixels_changed(self, text=None):
        txt = text or self.w["pixels"].currentText()
        self.chan.set_layout(txt)
        self._chan_changed()
        if getattr(self, "session", None) and self.w["src"].currentIndex() == 0:
            # tell the sensor (EVM GUI) to switch layout; the grid follows once the new channels arrive
            self.session.send_command(self._pixels_cmd())
            self.msg.setText(f"Sent {self._pixels_cmd()}: switching the sensor to {txt.replace(' x ', '×')}…")

    def _pixels_cmd(self):
        """EVM GUI command for the pixel layout: (P0) 3x3, (P1) 4x4, (P2) 3x6, (P3) 8x8 (as TMF8828PiDataReader)."""
        return f"(P{max(0, self.w['pixels'].currentIndex())})"

    def _layout_from_data(self, name, why):
        """The incoming data (or its #ITT row) needs another pixel layout: follow it and say so."""
        if self.w["pixels"].currentText() != name:
            self.w["pixels"].blockSignals(True)
            self.w["pixels"].setCurrentText(name)
            self.w["pixels"].blockSignals(False)
        self.msg.setText(why)

    def _sensor_mode_changed(self, text):
        preset_iters = {"Default": 550, "High Power": 4000, "Low Power": 50}
        if text in preset_iters:
            self.w["iters"].setValue(preset_iters[text])
            if getattr(self, "session", None) and self.w["src"].currentIndex() == 0:
                self.iters_changed()

    def run_sequence(self):
        if self.session:
            n = self.w["seq_count"].value()
            self.session.ctl.put(("seq", n))
            self.msg.setText(f"Triggering sequence of {n} measurements…")

    def sensor_calibrate(self):
        if self.session and self.w["src"].currentIndex() == 0:
            self.session.send_command("(C)")
            self.msg.setText("Sent sensor on-chip optical calibration command (C).")
        else:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "Sensor calibration requires an active TCP connection to the EVM GUI.\n"
                "Start acquisition first or click 'Calibrate' inside the EVM GUI."
            )

    def sensor_clear_calibrate(self):
        if self.session and self.w["src"].currentIndex() == 0:
            self.session.send_command("(C0)")
            self.msg.setText("Sent clear sensor calibration command (C0).")
        else:
            QtWidgets.QMessageBox.information(
                self, APP_NAME,
                "Sensor calibration requires an active TCP connection to the EVM GUI."
            )

    def take_snapshot(self):
        if not getattr(self, "last_frame", None):
            QtWidgets.QMessageBox.information(self, APP_NAME, "No acquisition frame to snapshot yet.")
            return
        ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        folder = self.persist.s.value("live/rec_dir", "") or os.path.expanduser("~")
        suggest = os.path.join(folder, f"snapshot_{ts}.csv")
        p, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Histogram Snapshot", suggest,
            "CSV files (*.csv);;PNG Image (*.png);;All files (*)"
        )
        if not p:
            return
        p = os.path.normpath(p)
        try:
            if p.lower().endswith(".png"):
                import pyqtgraph.exporters
                exporter = pyqtgraph.exporters.ImageExporter(self.p_h)
                exporter.export(p)
                self.msg.setText(f"Snapshot plot image saved: {os.path.basename(p)}")
            else:
                if not p.lower().endswith(".csv"):
                    p += ".csv"
                fr = self.last_frame
                with open(p, "w", newline="") as fh:
                    fh.write("sep=;\n")
                    fh.write(f"#SNAP;{time.time():.3f};frame_{self.n_frames}\n")
                    ch = fr.get("channels") or {}
                    for z, hist in sorted(ch.items()):
                        h_str = ";".join(str(int(round(x))) for x in hist)
                        fh.write(f"#HLONG{z:02d};{h_str}\n")
                    if fr.get("ref") is not None:
                        r_str = ";".join(str(int(round(x))) for x in fr["ref"])
                        fh.write(f"#HOCAL0;{r_str}\n")
                self.msg.setText(f"Snapshot histogram data saved: {os.path.basename(p)}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, APP_NAME, f"Could not save snapshot: {e}")

    def poll(self):
        if not self.session:
            return
        frame = None
        fitted = False
        while True:
            try:
                kind, p = self.session.out.get_nowait()
            except Exception:
                break
            if kind == "frame":
                frame = p
                self.n_frames += 1
                self.T_last = p["t_s"]
            elif kind == "result":
                fitted = True
                self.T.append(p["t_s"])
                self.MUA.append(p["mua"])
                self.MUSP.append(p["musp"])
                self.win_bins = p.get("window")
                self.last_fit = p
            elif kind == "status":
                self.msg.setText(p)
                if p == "connected":
                    self.status.emit(self._conn_text(), C["ok"])
                    self._update_sensor_badge()
            elif kind == "itt":
                self.itt = p
                self._update_sensor_badge(itt=p)
                mode_txt = (p.get("mode") or "").strip()
                pix = re.sub(r"\s*[x×X]\s*", " x ", mode_txt)
                if pix in ChannelPicker.LAYOUTS:
                    # the #ITT 'Mode' column is the zone layout (e.g. '4 x 4'). Follow it only when it changes and
                    # agrees with the channels actually arriving -- otherwise the data decides (no tug-of-war that
                    # rebuilds the grid every frame and swallows clicks).
                    new_itt = pix != getattr(self, "_itt_pix", None)
                    self._itt_pix = pix
                    if new_itt and self.chan.fits(pix) and self.w["pixels"].currentText() != pix:
                        self.w["pixels"].blockSignals(True)
                        self.w["pixels"].setCurrentText(pix)
                        self.w["pixels"].blockSignals(False)
                        self.chan.set_layout(pix)
                        self._chan_changed()
                        self.msg.setText(f"Pixels set to {pix.replace(' x ', '×')} from the data's #ITT row.")
                elif mode_txt:
                    idx = self.w["sensor_mode"].findText(mode_txt)
                    if idx < 0 and "high" in mode_txt.lower():
                        idx = self.w["sensor_mode"].findText("High Power")
                    if idx >= 0:
                        self.w["sensor_mode"].blockSignals(True)
                        self.w["sensor_mode"].setCurrentIndex(idx)
                        self.w["sensor_mode"].blockSignals(False)
                fov = self.SPAD_MAP_FOV.get(str(p.get("spad_map") or "").strip(), str(p.get("spad_map") or "").strip())
                if fov and not fov.isdigit():
                    idx = self.w["spad_map"].findText(fov)
                    if idx < 0:
                        self.w["spad_map"].addItem(fov)
                        idx = self.w["spad_map"].findText(fov)
                    self.w["spad_map"].blockSignals(True)
                    self.w["spad_map"].setCurrentIndex(idx)
                    self.w["spad_map"].blockSignals(False)
            elif kind == "telemetry":
                self.telemetry_data = p
            elif kind == "rec_started":
                self.rec_started = time.time()
                self.T, self.MUA, self.MUSP = [], [], []
                self.msg.setText("Recording → " + ", ".join(os.path.basename(x) for x in p))
                self.status.emit(self._conn_text() + " · ● REC", C["ok"])
            elif kind == "rec_stopped":
                self._rec_done(p)
            elif kind == "warning":
                self.msg.setText("⚠ " + p)
            elif kind == "error":
                self.msg.setText("✖ " + p)
                self.status.emit("Disconnected", C["bad"])
                QtWidgets.QMessageBox.critical(self, APP_NAME, p)
            elif kind == "stopped":
                self.finish()
                return
        if frame is not None:
            self.show_frame(frame)
        if fitted:
            self.redraw()
        d = self.w["dur"].value() if self.w["dur_on"].isChecked() else 0
        if self.recording and d and self.rec_started and time.time() - self.rec_started >= d and self.rec_btn.isEnabled():
            self.session.stop_recording()
            self.rec_btn.setEnabled(False)

    def finish(self):
        self.timer.stop()
        self.session = None
        if self.recording:
            self._rec_done([])
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.cmd_btn.setEnabled(False)
        self.rec_btn.setEnabled(False)
        self.seq_btn.setEnabled(False)
        self.sensor_cal_btn.setEnabled(False)
        self.sensor_clear_cal_btn.setEnabled(False)
        self.msg.setText(f"Acquisition stopped · {self.n_frames} frames acquired.")
        self.status.emit("Disconnected", C["bad"])
        self._update_sensor_badge()

    def _chan_changed(self):
        self.chan_box.set_title("Spatial Channels · " + self.chan.describe())
        if getattr(self, "last_frame", None) is not None:
            self.show_frame(self.last_frame, count=False)

    def show_frame(self, fr, count=True):
        self.last_frame = fr
        self._update_sensor_badge(frame=fr)
        ch = fr.get("channels") or {}
        if count:                                   # a new frame: check the grid matches the data's channels
            self.chan.ensure(ch.keys())
        sel = [c for c in self.chan.selected() if c in ch]
        for it in getattr(self, "c_over", []):
            self.p_h.removeItem(it)
        self.c_over = []
        n = len(fr["hist"])
        ctl = self.h_ctl
        norm, log = ctl.norm(), ctl.log()
        x = hist_x(n, ctl.unit())
        curve_m = ctl.curve_mode()
        has_dl = fr.get("has_dl", False)
        ch_raw = fr.get("channels_raw") or ch
        ch_floor = fr.get("channels_floor") or ch
        ch_clean = fr.get("channels_clean") or ch_floor
        ch_bg = fr.get("channels_bg") or {}

        self.c_raw.setData([], [])
        self.c_raw.setVisible(False)
        self.c_bg.setData([], [])
        self.c_bg.setVisible(False)
        self.c_clean.setData([], [])
        self.c_clean.setVisible(False)

        if curve_m == "raw":
            active_ch = ch_raw
            hist_def = fr.get("hist_raw", fr["hist"])
            pen_col = C["text"]
            brush_col = C["text"] + "20"
            mode_tag = "Raw counts"
        else:
            active_ch = ch_clean
            hist_def = fr.get("hist_clean", fr["hist"])
            pen_col = C["accent"]
            brush_col = C["accent"] + "30"
            mode_tag = "Background corrected (DL-sub)" if has_dl else "Background corrected"

        self.c_h.setPen(pg.mkPen(pen_col, width=1.5))
        self.c_h.setBrush(pg.mkBrush(brush_col))

        if not ctl.show_meas():
            self.c_h.setData([], [])
            self.c_h.setVisible(False)
            top = 1.0
        elif self.chan.overlay.isChecked() and len(sel) > 1:
            self.c_h.setData([], [])
            self.c_h.setVisible(False)
            top = 0.0
            for i, c in enumerate(sel):
                ch_data = active_ch.get(c, np.zeros(n))
                yc = hist_y(ch_data, norm, log)
                colr = pg.intColor(i, hues=max(len(sel), 2))
                it = self.p_h.plot(x, yc, pen=pg.mkPen(colr, width=1.3), symbol="o", symbolSize=4, symbolPen=None,
                                   symbolBrush=colr,
                                   name=f"ch {c}" if len(sel) <= 16 else None)
                self.c_over.append(it)
                top = max(top, np.nanmax(yc) if len(yc) else 0.0)
        else:
            self.c_h.setVisible(True)
            y = np.asarray(sum(active_ch[c] for c in sel if c in active_ch), dtype=float) if sel else np.asarray(hist_def, dtype=float)
            y = hist_y(y, norm, log)
            self.c_h.setFillLevel(None if log else 0)
            self.c_h.setData(x, y)
            top = np.nanmax(y) if len(y) else 1.0

        shown = (sum(float(np.asarray(ch_raw.get(c, ch.get(c, np.zeros(n))))[12:].sum()) for c in sel) if ch else fr.get("counts", 0))
        if fr.get("ref") is not None and ctl.show_ref():
            r = np.asarray(fr["ref"], dtype=float)
            sc = (top / np.nanmax(r)) if np.nanmax(r) > 0 and top > 0 else 1.0
            rr = r * sc
            if log:
                rr = np.clip(rr, top * 1e-5 if norm != "none" else 1.0, None)
            self.c_ref.setData(hist_x(len(r), ctl.unit()), rr)
            self.c_ref.setVisible(True)
        else:
            self.c_ref.setData([], [])
            self.c_ref.setVisible(False)

        wb = getattr(self, "win_bins", None)
        if self.w["fit_on"].isChecked() and wb:
            f = 1.0 if ctl.unit() == "bins" else V.DT_PS / 1000.0
            self.win.setRegion((wb[0] * f, wb[1] * f))
            self.win.show()
        else:
            self.win.hide()
        itt = getattr(self, "itt", None) or {}
        info = f"[{mode_tag}] · {self.chan.describe()} · frame {self.n_frames} · t = {fr['t_s']:.1f} s"
        if itt:
            info += f" · {itt.get('iterations_k', '?')}k pulses"
        if fr.get("recording"):
            info = "● REC  " + info
        self.p_h.setTitle(f"<span style='color:{C['bad'] if fr.get('recording') else C['muted']}'>{info}</span>",
                          size="9pt")
        ctl.apply_axes(self.p_h)
        self._update_legend()

        # Live throughput & arrival rate calculation
        now_wall = time.time()
        now_ts = fr.get("t_s", 0.0)
        self._frame_times.append((now_wall, now_ts))
        if len(self._frame_times) > 12:
            self._frame_times.pop(0)

        freq_hz = 0.0
        if len(self._frame_times) >= 2:
            ts_diffs = np.diff([t for _, t in self._frame_times])
            valid_ts = ts_diffs[(ts_diffs > 0.01) & (ts_diffs < 10.0)]
            if len(valid_ts) >= 2:
                dt_med = float(np.median(valid_ts))
                freq_hz = (1.0 / dt_med) if dt_med > 0 else 0.0
            else:
                wall_diffs = np.diff([w for w, _ in self._frame_times])
                valid_wall = wall_diffs[(wall_diffs > 0.01) & (wall_diffs < 10.0)]
                if len(valid_wall) >= 2:
                    dt_med = float(np.median(valid_wall))
                    freq_hz = (1.0 / dt_med) if dt_med > 0 else 0.0

        if freq_hz <= 0.0:
            mode_txt = str(itt.get("mode", "") or getattr(self, "itt", {}).get("mode", ""))
            freq_hz = 4.06 if "3 x 3" in mode_txt else (1.0 / getattr(V, "FRAME_PERIOD_S", 0.46))

        self._last_fps = freq_hz
        chans_dict = fr.get("channels") or {}
        n_chans = len(chans_dict) if chans_dict else (16 if "4 x 4" in str(itt) else 9)
        curves_per_s = n_chans * freq_hz

        self.k_rate.set(f"{freq_hz:.2f} Hz", f"{curves_per_s:.0f} curves/s · {self.n_frames:,} frames")
        self.k_rate.setToolTip(
            f"Acquisition Frequency & Throughput:\n"
            f"• Frame rate: {freq_hz:.2f} Hz (1 full frame every {1.0/freq_hz:.2f} s)\n"
            f"• Per-channel curve rate: {freq_hz:.2f} curves/s (identical for all channels)\n"
            f"• Total throughput: {curves_per_s:.1f} curves/s across {n_chans} channels\n"
            f"• Total frames captured: {self.n_frames:,} frames (t = {fr.get('t_s', 0.0):.1f} s)\n"
            f"Note on 4×4 mode: Raw sensor packets arrive at ~4.06 Hz, but each packet only updates 8 zones "
            f"(the other 8 are repeated). A complete 16-channel frame requires two packets, yielding a genuine frame rate of ~2.17 Hz."
        )

        # Compute peak bin count across active channels
        peak_val = 0.0
        if sel and ch_raw:
            arrs = [np.asarray(ch_raw.get(c, ch.get(c, np.zeros(n))), dtype=float) for c in sel]
            if arrs:
                peak_val = float(np.nanmax(sum(arrs)))
        elif len(fr.get("hist", [])):
            peak_val = float(np.nanmax(fr["hist"]))

        lf = getattr(self, "last_fit", None)
        if self.w["fit_on"].isChecked() and lf:
            m, s = lf["mua"], lf["musp"]
            mode = self.w["mode"].currentIndex()
            self.k_mua.set(f"{m:.4f}" if np.isfinite(m) else "—",
                           ("cm⁻¹ · fixed" if mode == 2 else "cm⁻¹") if np.isfinite(m) else "below signal threshold")
            self.k_musp.set(f"{s:.2f}" if np.isfinite(s) else "—", "cm⁻¹ · fixed" if mode == 1 else "cm⁻¹")
            self.k_dt.set(f"{lf['dT_ps']:.0f}", "ps vs. HOCAL")
            self.k_I.set(f"{shown:,.0f}", f"Peak: {peak_val:,.0f} · I = {lf['intensity']:.4g}/k pulses")
        else:
            self.k_I.set(f"{shown:,.0f}", f"Peak: {peak_val:,.0f} counts/bin (shown channels)")
        self.k_I.setToolTip(
            f"Total Detected Photon Counts:\n"
            f"• Integrated counts: {shown:,.0f} photons across active channels\n"
            f"• Peak bin count: {peak_val:,.0f} photons in single time bin\n"
            f"• Fit window intensity: {lf['intensity']:.4g} / k pulses" if (lf and self.w["fit_on"].isChecked()) else
            f"Total Detected Photon Counts:\n• Integrated counts: {shown:,.0f} photons\n• Peak bin count: {peak_val:,.0f} photons"
        )
        if hasattr(self, "opt_window") and self.opt_window:
            self.opt_window.update_kpis(lf if self.w["fit_on"].isChecked() else None, fr, self.n_frames)

    def _update_legend(self):
        if not hasattr(self.p_h, "legend") or not self.p_h.legend:
            return
        leg = self.p_h.legend
        leg.clear()
        if not self.h_ctl.legend_cb.isChecked():
            leg.setVisible(False)
            return
        leg.setVisible(True)
        curve_m = self.h_ctl.curve_mode()
        if self.c_h.isVisible():
            leg_name = "raw counts" if curve_m == "raw" else "BG corrected"
            leg.addItem(self.c_h, leg_name)
            for it in getattr(self, "c_over", []):
                name = getattr(it, "_legend_name", None) or (it.opts.get("name") if hasattr(it, "opts") else None)
                if name:
                    leg.addItem(it, name)
        if self.c_ref.isVisible() and self.h_ctl.show_ref():
            leg.addItem(self.c_ref, "HOCAL")

    def redraw(self, *_):
        t = np.asarray(self.T, dtype=float)
        self.c_mua.setData(t, np.asarray(self.MUA, dtype=float))
        self.c_musp.setData(t, np.asarray(self.MUSP, dtype=float))

    def shutdown(self):
        self.stop()
        if hasattr(self, "opt_window") and self.opt_window:
            self.opt_window.force_close()


