"""Data Viewer tab: interactive replay scrubbing and results review."""
import csv
import datetime as dt
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import pyqtgraph as pg

import figures as FIG
import tmf_moments_solver as V
from gui.styles import (
    C, CHANNEL_COLORS, APP_NAME, _PG_SYM, card, subcard, muted,
    pill, set_pill, style_plot, CountAxisItem
)
from gui.persistence import (
    Persist, widget_values, set_widget_values
)
from gui.widgets import (
    PathEdit, Collapsible, UnitForm, form_layout, dspin, ispin, unit_box,
    HistControls, PlotTools, plot_bar, with_bar, TabStack, scroll,
    is_recording, read_values_file, base, hist_x, hist_y, new_table, Kpi,
    ChannelPicker
)

class DataViewerTab(QtWidgets.QWidget):
    def __init__(self, persist, main_win=None):
        super().__init__()
        self.persist = persist
        self.main_win = main_win

        # --- Replay state ---
        self.rep_frames = {}       # {zone_str: 2D_array (n_frames, 128)}
        self.rep_times = []        # array of times in seconds
        self.rep_iters = None
        self.rep_cur_frame = 0
        self.rep_playing = False
        self.rep_zone_map = {}     # {int_ch: zone_str}
        self.rep_chan_map = {}     # {zone_str: int_ch}
        self.rep_timer = QtCore.QTimer(self)
        self.rep_timer.timeout.connect(self._on_replay_tick)

        # --- Results / Validation state ---
        self.res_records = []      # list of dicts: {frame, t_s, dT_ps, intensity, mua, mua_sd, musp, musp_sd}
        self.res_file = None
        self.cmp_rows = []

        # --- Main Layout ---
        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(16, 14, 16, 16)
        v.setSpacing(10)

        # Mode switcher header
        hdr = QtWidgets.QWidget()
        hl = QtWidgets.QHBoxLayout(hdr)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.setSpacing(10)

        self.mode_grp = QtWidgets.QButtonGroup(self)
        self.btn_replay_mode = QtWidgets.QPushButton("Replay Past Recording")
        self.btn_replay_mode.setObjectName("modeBtn")
        self.btn_replay_mode.setCheckable(True)
        self.btn_replay_mode.setChecked(True)
        self.btn_replay_mode.setToolTip("Replay and step through raw pToF histogram frames from any past recording.")

        self.btn_results_mode = QtWidgets.QPushButton("View Results && Output Validation")
        self.btn_results_mode.setObjectName("modeBtn")
        self.btn_results_mode.setCheckable(True)
        self.btn_results_mode.setChecked(False)
        self.btn_results_mode.setToolTip("Inspect optical property time series, summary statistics, and validate against expected values (y = x plot).")

        self.mode_grp.addButton(self.btn_replay_mode, 0)
        self.mode_grp.addButton(self.btn_results_mode, 1)
        self.btn_replay_mode.clicked.connect(lambda: self.sub_stack.setCurrentIndex(0))
        self.btn_results_mode.clicked.connect(lambda: self.sub_stack.setCurrentIndex(1))

        hl.addWidget(self.btn_replay_mode)
        hl.addWidget(self.btn_results_mode)
        hl.addStretch(1)

        self.hdr_pill = pill("Ready", C["muted"])
        hl.addWidget(self.hdr_pill, 0, Qt.AlignVCenter)
        v.addWidget(hdr)

        # Stacked view: Option 1 vs Option 2
        self.sub_stack = QtWidgets.QStackedWidget()
        v.addWidget(self.sub_stack, 1)

        self.page_replay = self._build_replay_ui()
        self.sub_stack.addWidget(self.page_replay)

        self.page_results = self._build_results_ui()
        self.sub_stack.addWidget(self.page_results)

    # =========================================================================
    # OPTION 1: REPLAY PAST RECORDING
    # =========================================================================
    def _build_replay_ui(self):
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(16)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(12)

        # 1. Recording Source
        c_src, l_src = card("1 · Recording Source")
        self.rep_path = PathEdit("file", "CSV files (*.csv);;All files (*)", "Select raw pToF recording (.csv)…")
        self.rep_path.changed.connect(self._load_replay_file)
        l_src.addWidget(self.rep_path)

        self.sc_rep_info, scl = subcard()
        self.lbl_rep_info = QtWidgets.QLabel("No recording loaded. Choose a .csv file above to begin replay.")
        self.lbl_rep_info.setWordWrap(True)
        self.lbl_rep_info.setObjectName("muted")
        scl.addWidget(self.lbl_rep_info)
        l_src.addWidget(self.sc_rep_info)
        lv.addWidget(c_src)

        # 2. Playback Controls
        c_ctrl, l_ctrl = card("2 · Playback Controls")

        # Scrubber row
        scrub_row = QtWidgets.QHBoxLayout()
        self.rep_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.rep_slider.setRange(0, 0)
        self.rep_slider.setEnabled(False)
        self.rep_slider.valueChanged.connect(self._on_scrubber_moved)
        self.lbl_rep_frame = QtWidgets.QLabel("Frame 0 / 0  ·  0.00 s")
        self.lbl_rep_frame.setObjectName("muted")
        scrub_row.addWidget(self.rep_slider, 1)
        scrub_row.addWidget(self.lbl_rep_frame)
        l_ctrl.addLayout(scrub_row)

        # Button row 1: Play, Stop, Steps
        hb_btns = QtWidgets.QHBoxLayout()
        hb_btns.setSpacing(6)
        self.btn_rep_play = QtWidgets.QPushButton("▶  Play")
        self.btn_rep_play.setObjectName("primary")
        self.btn_rep_play.setEnabled(False)
        self.btn_rep_play.clicked.connect(self._toggle_replay_play)

        self.btn_rep_stop = QtWidgets.QPushButton("■  Stop")
        self.btn_rep_stop.setEnabled(False)
        self.btn_rep_stop.clicked.connect(self._stop_replay)

        self.btn_rep_prev = QtWidgets.QPushButton("◀")
        self.btn_rep_prev.setToolTip("Step backward 1 frame")
        self.btn_rep_prev.setEnabled(False)
        self.btn_rep_prev.clicked.connect(lambda: self._step_frame(-1))

        self.btn_rep_next = QtWidgets.QPushButton("▶")
        self.btn_rep_next.setToolTip("Step forward 1 frame")
        self.btn_rep_next.setEnabled(False)
        self.btn_rep_next.clicked.connect(lambda: self._step_frame(1))

        hb_btns.addWidget(self.btn_rep_play, 2)
        hb_btns.addWidget(self.btn_rep_stop, 1)
        hb_btns.addWidget(self.btn_rep_prev, 1)
        hb_btns.addWidget(self.btn_rep_next, 1)
        l_ctrl.addLayout(hb_btns)

        # Speed and Loop row
        h_spd = QtWidgets.QHBoxLayout()
        h_spd.addWidget(QtWidgets.QLabel("Speed:"))
        self.rep_speed_cb = QtWidgets.QComboBox()
        self.rep_speed_cb.addItems(["0.25x", "0.5x", "1.0x (Real-time)", "2.0x", "5.0x", "10.0x", "Max (20 fps)"])
        self.rep_speed_cb.setCurrentIndex(2)
        self.rep_speed_cb.currentIndexChanged.connect(self._update_replay_speed)
        h_spd.addWidget(self.rep_speed_cb, 1)

        self.rep_loop_cb = QtWidgets.QCheckBox("Loop")
        self.rep_loop_cb.setChecked(True)
        self.rep_loop_cb.setToolTip("Restart playback from frame 1 when reaching the end")
        h_spd.addWidget(self.rep_loop_cb)
        l_ctrl.addLayout(h_spd)
        lv.addWidget(c_ctrl)

        # 3. Channel & Histogram Controls
        c_chan, l_chan = card("3 · Channel & Histogram Display")
        self.chan = ChannelPicker()
        self.chan.overlay.setChecked(False)  # Summed by default as requested
        self.chan.overlay.setText("Overlay individual channels (otherwise summed)")
        self.chan.changed.connect(self._render_replay_frame)
        l_chan.addWidget(self.chan)

        self.rep_hctl = HistControls()
        self.rep_hctl.ref_cb.setVisible(False)  # hide ref checkbox in replay
        self.rep_hctl.changed.connect(lambda: (self.rep_hctl.apply_axes(self.p_rep), self._render_replay_frame()))
        l_chan.addWidget(self.rep_hctl)
        lv.addWidget(c_chan)

        # 4. Instant Frame Metrics
        c_kpi, l_kpi = card("4 · Current Frame Metrics")
        hk = QtWidgets.QHBoxLayout()
        hk.setSpacing(8)
        self.kpi_peak = Kpi("Peak Counts", C["accent"])
        self.kpi_arrival = Kpi("Mean Arrival ⟨t⟩", C["dt"], "ns")
        self.kpi_total = Kpi("Sum of Counts", C["inten"])
        hk.addWidget(self.kpi_peak)
        hk.addWidget(self.kpi_arrival)
        hk.addWidget(self.kpi_total)
        l_kpi.addLayout(hk)
        lv.addWidget(c_kpi)

        lv.addStretch(1)

        h.addWidget(scroll(left, 450))

        # Right side: Plot
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        self.gw_rep = pg.GraphicsLayoutWidget()
        self.gw_rep.setBackground(C["card"])
        self.p_rep = style_plot(self.gw_rep.addPlot(row=0, col=0, axisItems={"left": CountAxisItem(orientation="left")}), "photon counts", "time (ns)")
        self.p_rep.setTitle(f"<span style='color:{C['text']}; font-weight:600;'>pToF Histogram Replay</span>", size="10.5pt")
        self.p_rep.addLegend(offset=(-10, 10))

        self.rep_curves = {}
        self.rep_hctl.apply_axes(self.p_rep)

        self.rep_tools = PlotTools(self.gw_rep, [("replay_histogram", self.p_rep)], "replay_histogram")
        rv.addWidget(with_bar(plot_bar([self.rep_hctl], self.rep_tools), self.gw_rep), 1)

        h.addWidget(right, 1)
        return page

    # =========================================================================
    # OPTION 2: VIEW RESULTS & OUTPUT VALIDATION
    # =========================================================================
    def _build_results_ui(self):
        page = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(page)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(16)

        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 6, 0)
        lv.setSpacing(12)

        # 1. Results Source
        c_src, l_src = card("1 · Results & Recording Selection")
        self.res_path = PathEdit("file", "CSV files (*.csv);;All files (*)", "Select results (*_timeseries.csv, *_live_results.csv) or recording (.csv)…")
        self.res_path.changed.connect(self._load_results_file)
        l_src.addWidget(self.res_path)

        self.lbl_res_note = muted("Select a precomputed timeseries file or a raw recording with companion results.")
        self.lbl_res_note.setWordWrap(True)
        l_src.addWidget(self.lbl_res_note)

        # Bridge button: Open in Offline Analysis
        self.btn_bridge_off = QtWidgets.QPushButton("Open in Offline Analysis →")
        self.btn_bridge_off.setObjectName("ghost")
        self.btn_bridge_off.setToolTip("Send this recording/dataset to the Offline Analysis tab for full batch processing and report generation.")
        self.btn_bridge_off.clicked.connect(self.open_in_offline)
        l_src.addWidget(self.btn_bridge_off)
        lv.addWidget(c_src)

        # 2. Output Validation
        c_val, l_val = card("2 · Output Validation")
        self.chk_val_enable = QtWidgets.QCheckBox("Validate against expected values")
        self.chk_val_enable.setChecked(True)
        self.chk_val_enable.setToolTip("Enable validation comparison against certified phantom values (y = x with standard ±10 % band).")
        self.chk_val_enable.toggled.connect(self._render_results_plots)
        l_val.addWidget(self.chk_val_enable)

        self.sc_val, scl_val = subcard()
        fv = form_layout()

        self.val_mua_exp = dspin(0.001, 10.0, 0.200, 3, 0.01, "cm⁻¹")
        self.val_mua_exp.setToolTip("Expected certified absorption coefficient μa")
        self.val_mua_exp.valueChanged.connect(self._render_results_plots)
        fv.addRow("Expected μa", self.val_mua_exp)

        self.val_musp_exp = dspin(0.1, 100.0, 10.00, 2, 0.5, "cm⁻¹")
        self.val_musp_exp.setToolTip("Expected certified reduced scattering coefficient μs′")
        self.val_musp_exp.valueChanged.connect(self._render_results_plots)
        fv.addRow("Expected μs′", self.val_musp_exp)

        scl_val.addLayout(fv)

        self.val_compare_file = PathEdit("file", "CSV (*.csv);;All files (*)", "Or load multi-phantom compare.csv…")
        self.val_compare_file.changed.connect(self._load_compare_file)
        scl_val.addWidget(self.val_compare_file)

        b_tmpl = QtWidgets.QPushButton("Download CSV Template…")
        b_tmpl.setObjectName("ghost")
        b_tmpl.setToolTip("Download expected values comparison template CSV")
        b_tmpl.clicked.connect(self._download_compare_template)
        scl_val.addWidget(b_tmpl)

        lbl_band_info = muted("Output validation: recovered properties evaluated against expected certified values with a standard ±10 % tolerance band (green).")
        lbl_band_info.setWordWrap(True)
        scl_val.addWidget(lbl_band_info)

        l_val.addWidget(self.sc_val)
        lv.addWidget(c_val)

        # 3. Summary & Validation Status
        c_sum, l_sum = card("3 · Summary & Validation Status")
        hk = QtWidgets.QHBoxLayout()
        hk.setSpacing(8)
        self.kpi_mua = Kpi("Recovered μa", C["mua"], "cm⁻¹")
        self.kpi_musp = Kpi("Recovered μs′", C["musp"], "cm⁻¹")
        hk.addWidget(self.kpi_mua)
        hk.addWidget(self.kpi_musp)
        l_sum.addLayout(hk)

        self.sc_val_status, scl_val_st = subcard()
        self.lbl_val_mua = QtWidgets.QLabel("Δμa: —")
        self.lbl_val_musp = QtWidgets.QLabel("Δμs′: —")
        self.pill_val = pill("Validation: —", C["muted"])

        scl_val_st.addWidget(self.lbl_val_mua)
        scl_val_st.addWidget(self.lbl_val_musp)
        scl_val_st.addWidget(self.pill_val)
        l_sum.addWidget(self.sc_val_status)

        lv.addWidget(c_sum)
        lv.addStretch(1)

        h.addWidget(scroll(left, 450))

        # Right side: Tabbed Views
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)

        self.res_tabs = TabStack()

        # Action buttons in tab header
        self.btn_res_csv = QtWidgets.QPushButton("Export CSV…")
        self.btn_res_csv.setEnabled(False)
        self.btn_res_csv.clicked.connect(self._export_results_csv)
        self.res_tabs.bar_row.addWidget(self.btn_res_csv, 0, Qt.AlignVCenter)

        # Tab 1: Time Series
        t_ts = QtWidgets.QWidget()
        tl_ts = QtWidgets.QVBoxLayout(t_ts)
        tl_ts.setContentsMargins(8, 8, 8, 8)
        tl_ts.setSpacing(8)

        # Time series controls toolbar
        ts_bar = QtWidgets.QHBoxLayout()
        ts_bar.addWidget(QtWidgets.QLabel("Moving Average:"))
        self.spin_ts_roll = dspin(0, 3600, 0, 1, 1, "s")
        self.spin_ts_roll.setToolTip("Moving average window for smoothing (0 = unsmoothed)")
        self.spin_ts_roll.valueChanged.connect(self._render_results_plots)
        ts_bar.addWidget(unit_box(self.spin_ts_roll), 0)

        self.chk_ts_err = QtWidgets.QCheckBox("Show ±1 SD error bars")
        self.chk_ts_err.toggled.connect(self._render_results_plots)
        ts_bar.addWidget(self.chk_ts_err)
        ts_bar.addStretch(1)
        tl_ts.addLayout(ts_bar)

        self.gw_ts = pg.GraphicsLayoutWidget()
        self.gw_ts.setBackground(C["card"])
        self.p_ts_mua = style_plot(self.gw_ts.addPlot(row=0, col=0), "μa (cm⁻¹)")
        self.p_ts_musp = style_plot(self.gw_ts.addPlot(row=1, col=0), "μs′ (cm⁻¹)", "time (s)")
        self.p_ts_musp.setXLink(self.p_ts_mua)
        self.p_ts_mua.getAxis("bottom").setStyle(showValues=False)

        sym = dict(symbol="o", symbolSize=5, symbolPen=None)
        self.c_ts_mua = self.p_ts_mua.plot(pen=pg.mkPen(C["mua"], width=1.6), symbolBrush=C["mua"], connect="finite", name="μa recovered", **sym)
        self.c_ts_musp = self.p_ts_musp.plot(pen=pg.mkPen(C["musp"], width=1.6), symbolBrush=C["musp"], connect="finite", name="μs′ recovered", **sym)
        self.line_exp_mua = self.p_ts_mua.plot(pen=pg.mkPen(C["mua"], width=1.4, style=Qt.DashLine), name="μa expected")
        self.line_exp_musp = self.p_ts_musp.plot(pen=pg.mkPen(C["musp"], width=1.4, style=Qt.DashLine), name="μs′ expected")
        self.p_ts_mua.addLegend(offset=(10, 10))
        self.p_ts_musp.addLegend(offset=(10, 10))

        self.tools_ts = PlotTools(self.gw_ts, [("ts_mua", self.p_ts_mua), ("ts_musp", self.p_ts_musp)], "results_timeseries")
        tl_ts.addWidget(with_bar(plot_bar([], self.tools_ts), self.gw_ts), 1)
        self.res_tabs.addTab(t_ts, "Time Series")

        # Tab 2: Output Validation Plot
        t_val = QtWidgets.QWidget()
        tl_val = QtWidgets.QVBoxLayout(t_val)
        tl_val.setContentsMargins(8, 8, 8, 8)
        tl_val.setSpacing(6)

        self.gw_val = pg.GraphicsLayoutWidget()
        self.gw_val.setBackground("w")  # Clean white background for publication validation plot
        self.p_val_mua = self._setup_val_plot(self.gw_val.addPlot(row=0, col=0), "μa")
        self.p_val_musp = self._setup_val_plot(self.gw_val.addPlot(row=0, col=1), "μs′")

        self.tools_val = PlotTools(self.gw_val, [("val_mua", self.p_val_mua), ("val_musp", self.p_val_musp)], "output_validation")
        tl_val.addWidget(with_bar(plot_bar([], self.tools_val), self.gw_val), 1)
        self.res_tabs.addTab(t_val, "Output Validation")

        # Tab 3: Data Table
        t_tbl = QtWidgets.QWidget()
        tl_tbl = QtWidgets.QVBoxLayout(t_tbl)
        tl_tbl.setContentsMargins(8, 8, 8, 8)
        tl_tbl.setSpacing(6)
        self.res_table = new_table()
        tl_tbl.addWidget(self.res_table, 1)
        self.res_tabs.addTab(t_tbl, "Data Table")

        rv.addWidget(self.res_tabs, 1)
        h.addWidget(right, 1)
        return page

    def _setup_val_plot(self, p, lab):
        ink = FIG.INK
        for ax in ("left", "bottom"):
            a = p.getAxis(ax)
            a.setPen(pg.mkPen(ink, width=1))
            a.setTextPen(ink)
            a.setStyle(tickLength=5)
            a.enableAutoSIPrefix(False)
        p.getAxis("left").setGrid(40)
        p.getAxis("bottom").setGrid(40)
        p.setLabel("left", f"recovered {lab} (cm⁻¹)", color=ink)
        p.setLabel("bottom", f"expected {lab} (cm⁻¹)", color=ink)
        p.setTitle(f"<span style='color:{ink}; font-size:12pt; font-weight:600;'>{lab}: recovered vs expected</span>", justify="left")
        p.addLegend(offset=(12, 6), labelTextColor=ink, brush=None, pen=None, labelTextSize="9pt")
        return p

    # =========================================================================
    # REPLAY LOGIC
    # =========================================================================
    def _load_replay_file(self, path):
        if not path or not os.path.isfile(path):
            return
        try:
            if not is_recording(path):
                self.lbl_rep_info.setText(
                    f"<b>{os.path.basename(path)}</b> is not a raw pToF histogram recording.<br>"
                    f"<span style='color:{C['warn']}'>This file does not contain raw #HLONG histogram rows. "
                    f"To replay frames, please select an EVM GUI recording CSV (e.g. sample.csv).<br><br>"
                    f"<i>Tip: To inspect fitted optical properties or time-series curves, switch to "
                    f"<b>'View Results & Output Validation'</b> mode above.</i></span>"
                )
                self.btn_rep_play.setEnabled(False)
                self.btn_rep_stop.setEnabled(False)
                self.btn_rep_prev.setEnabled(False)
                self.btn_rep_next.setEnabled(False)
                self.rep_slider.setEnabled(False)
                set_pill(self.hdr_pill, "Invalid File", C["bad"])
                return

            frames, times, iters = V.read_hlong_frames(path)
            if not frames:
                self.lbl_rep_info.setText(
                    f"<b>{os.path.basename(path)}</b> contains no valid histogram frames.<br>"
                    f"<span style='color:{C['warn']}'>Please select a raw EVM GUI recording CSV with #HLONG data.</span>"
                )
                self.btn_rep_play.setEnabled(False)
                self.btn_rep_stop.setEnabled(False)
                self.btn_rep_prev.setEnabled(False)
                self.btn_rep_next.setEnabled(False)
                self.rep_slider.setEnabled(False)
                set_pill(self.hdr_pill, "No Frames", C["bad"])
                return

            self.rep_frames = frames
            self.rep_times = times
            self.rep_iters = iters
            n_frames = len(next(iter(frames.values())))

            # Check for companion DL file in recording folder
            self.rep_dl_zones = None
            d = os.path.dirname(path)
            for cand in (os.path.join(d, "dl.csv"), os.path.join(d, "DL.csv"), os.path.join(d, "dl", "dl.csv")):
                if os.path.isfile(cand):
                    try:
                        self.rep_dl_zones, _ = V.read_dl(cand)
                        break
                    except Exception:
                        pass

            # Setup spatial measurement channels (#HLONG only)
            # Exclude #HOCAL - HOCAL is the internal optical reference pulse (IRF / HOCAL), not a spatial channel!
            hlong_keys = [z for z in sorted(frames.keys()) if z.startswith("#HLONG")]
            meas_zones = hlong_keys if hlong_keys else sorted(frames.keys())

            # Store reference pulse (HOCAL or companion IRF)
            hocal_keys = [z for z in sorted(frames.keys()) if z.startswith("#HOCAL")]
            self.rep_hocal_data = None
            if hocal_keys:
                self.rep_hocal_data = [
                    V.clean_hocal(np.mean([frames[k][i] for k in hocal_keys], axis=0))
                    for i in range(n_frames)
                ]
                self.rep_hctl.ref_cb.setVisible(True)
            else:
                self.rep_hctl.ref_cb.setVisible(False)

            self.rep_zone_map = {}
            self.rep_chan_map = {}
            raw_nums = []
            for z in meas_zones:
                m = re.findall(r"\d+", z)
                if m:
                    raw_nums.append(int(m[0]))
            is_zero_based = (0 in raw_nums and len(meas_zones) not in raw_nums)
            for i, z in enumerate(meas_zones):
                m = re.findall(r"\d+", z)
                ch = (int(m[0]) + 1 if is_zero_based else int(m[0])) if m else (i + 1)
                self.rep_zone_map[ch] = z
                self.rep_chan_map[z] = ch

            self.chan.blockSignals(True)
            self.chan.ensure(self.rep_zone_map.keys())
            self.chan.overlay.setChecked(False)  # default to summed
            self.chan.blockSignals(False)

            # Setup scrubber
            self.rep_slider.blockSignals(True)
            self.rep_slider.setRange(0, n_frames - 1)
            self.rep_slider.setValue(0)
            self.rep_slider.setEnabled(True)
            self.rep_slider.blockSignals(False)
            self.rep_cur_frame = 0

            # Enable playback buttons
            self.btn_rep_play.setEnabled(True)
            self.btn_rep_stop.setEnabled(True)
            self.btn_rep_prev.setEnabled(True)
            self.btn_rep_next.setEnabled(True)

            dur_s = float(times[-1] - times[0]) if len(times) > 1 and np.isfinite(times[-1]) else 0.0
            fps = (n_frames / dur_s) if dur_s > 0 else 0.0
            layout_str = f" ({self.chan.layout_name})" if getattr(self.chan, "layout_name", None) else ""
            self.lbl_rep_info.setText(
                f"<b>{os.path.basename(path)}</b><br>"
                f"Frames: <b>{n_frames}</b>  ·  Duration: <b>{dur_s:.2f} s</b> ({fps:.1f} fps)<br>"
                f"Channels: <b>{len(meas_zones)}</b>{layout_str}  ·  Iterations: <b>{iters or 'default'}</b>"
            )
            set_pill(self.hdr_pill, f"Replay: {n_frames} frames", C["ok"])
            self._render_replay_frame()
        except Exception as e:
            self.lbl_rep_info.setText(f"Error loading recording: {e}")

    def _on_scrubber_moved(self, val):
        self.rep_cur_frame = val
        self._render_replay_frame()

    def _step_frame(self, delta):
        n_frames = len(self.rep_times) if len(self.rep_times) > 0 else 0
        if n_frames == 0:
            return
        nxt = (self.rep_cur_frame + delta)
        if nxt >= n_frames:
            if self.rep_loop_cb.isChecked():
                nxt = 0
            else:
                self._stop_replay()
                return
        elif nxt < 0:
            nxt = n_frames - 1
        self.rep_slider.setValue(nxt)

    def _toggle_replay_play(self):
        if self.rep_playing:
            self._pause_replay()
        else:
            self._start_replay()

    def _start_replay(self):
        self.rep_playing = True
        self.btn_rep_play.setText("⏸  Pause")
        self._update_replay_speed()
        self.rep_timer.start()

    def _pause_replay(self):
        self.rep_playing = False
        self.btn_rep_play.setText("▶  Play")
        self.rep_timer.stop()

    def _stop_replay(self):
        self._pause_replay()
        self.rep_slider.setValue(0)

    def _update_replay_speed(self):
        txt = self.rep_speed_cb.currentText()
        if "0.25x" in txt:
            interval = 400
        elif "0.5x" in txt:
            interval = 200
        elif "1.0x" in txt:
            interval = 100
        elif "2.0x" in txt:
            interval = 50
        elif "5.0x" in txt:
            interval = 25
        elif "10.0x" in txt:
            interval = 15
        else:
            interval = 50  # 20 fps
        self.rep_timer.setInterval(interval)

    def _on_replay_tick(self):
        self._step_frame(1)

    def _render_replay_frame(self):
        if not self.rep_frames:
            return
        f_idx = self.rep_cur_frame
        n_frames = len(self.rep_times)
        t_val = self.rep_times[f_idx] if f_idx < len(self.rep_times) and np.isfinite(self.rep_times[f_idx]) else f_idx * 0.1
        self.lbl_rep_frame.setText(f"Frame {f_idx + 1} / {n_frames}  ·  {t_val:.2f} s")

        norm = ("peak" if self.rep_hctl.norm_cb.currentIndex() == 1 else
                "area" if self.rep_hctl.norm_cb.currentIndex() == 2 else "none")
        unit = "bins" if self.rep_hctl.x_cb.currentIndex() == 1 else "ns"
        log = self.rep_hctl.log_cb.isChecked()

        sel = self.chan.selected()
        sel_zones = [self.rep_zone_map[c] for c in sel if c in self.rep_zone_map]

        # Clear existing curves
        for c in self.rep_curves.values():
            self.p_rep.removeItem(c)
        self.rep_curves.clear()

        if not sel_zones:
            self.kpi_peak.set("—")
            self.kpi_arrival.set("—")
            self.kpi_total.set("—")
            if hasattr(self.p_rep, "legend") and self.p_rep.legend is not None:
                self.p_rep.legend.clear()
                self.p_rep.legend.hide()
            self.p_rep.setTitle(
                f"<span style='color:{C['text']}; font-weight:600;'>pToF Histogram Replay</span> "
                f"<span style='color:{C['warn']}; font-size:9pt;'>[No channels selected]</span>",
                size="10.5pt"
            )
            return

        # Compute instant metrics on the sum of all selected channels
        raw_sum = sum(np.asarray(self.rep_frames[z][f_idx], dtype=float) for z in sel_zones)
        peak_val = int(np.nanmax(raw_sum))
        tot_photons = int(np.nansum(raw_sum))
        bins = np.arange(len(raw_sum))
        denom = np.nansum(raw_sum)
        mean_arrival_ns = float(np.nansum(bins * raw_sum) / denom * V.DT_PS / 1000.0) if denom > 0 else 0.0

        self.kpi_peak.set(f"{peak_val:,}")
        self.kpi_arrival.set(f"{mean_arrival_ns:.2f}")
        self.kpi_total.set(f"{tot_photons:,}")

        curve_m = self.rep_hctl.curve_mode()
        has_dl = bool(getattr(self, "rep_dl_zones", None))
        if curve_m == "raw":
            mode_tag = "Raw counts"
        else:
            mode_tag = "Background corrected" + (" (DL-sub)" if has_dl else "")

        def _get_curve_versions(raw_vec, zone_key):
            v = np.asarray(raw_vec, dtype=float)
            floor = np.median(v[:12]) if len(v) > 12 else 0.0
            dl_z = getattr(self, "rep_dl_zones", {}).get(zone_key) if getattr(self, "rep_dl_zones", None) else None
            if dl_z is not None:
                try:
                    c_z = V.clean_zone(v, dl_z, self.rep_iters)
                except Exception:
                    c_z = v - floor
            else:
                c_z = v - floor
            return v, c_z

        active_keys = set()
        sum_lbl = ""
        meas_top = 0.0

        # Render curves
        if self.chan.overlay.isChecked() and len(sel_zones) > 1:
            # Multi-channel overlay (reuse existing curve items for maximum FPS)
            for i, z in enumerate(sel_zones):
                raw_h = self.rep_frames[z][f_idx]
                r_v, c_v = _get_curve_versions(raw_h, z)
                to_plot = r_v if curve_m == "raw" else c_v
                xx = hist_x(len(to_plot), unit)
                yy = hist_y(to_plot, norm, log)
                meas_top = max(meas_top, float(np.nanmax(yy)) if len(yy) else 0.0)
                col = CHANNEL_COLORS[i % len(CHANNEL_COLORS)]
                ch_num = self.rep_chan_map.get(z, i + 1)
                curve_name = f"Ch {ch_num}"

                if z in self.rep_curves:
                    c = self.rep_curves[z]
                    c.setData(xx, yy)
                    c.setVisible(True)
                else:
                    c = self.p_rep.plot(
                        xx, yy,
                        pen=pg.mkPen(col, width=1.5),
                        symbol="o", symbolSize=4,
                        symbolPen=None, symbolBrush=col,
                        name=curve_name
                    )
                    self.rep_curves[z] = c
                active_keys.add(z)
        else:
            # Summed across all selected channels (or single channel)
            n_bins = len(next(iter(self.rep_frames.values()))[f_idx])
            sum_r = sum(np.asarray(self.rep_frames[z][f_idx], dtype=float) for z in sel_zones)
            sum_c = np.zeros(n_bins, dtype=float)
            for z in sel_zones:
                r_z, c_z = _get_curve_versions(self.rep_frames[z][f_idx], z)
                sum_c += c_z

            xx = hist_x(n_bins, unit)
            sum_lbl = f"Summed ({len(sel_zones)} ch)" if len(sel_zones) > 1 else f"Ch {self.rep_chan_map.get(sel_zones[0], 1)}"
            to_plot = sum_r if curve_m == "raw" else sum_c
            pen_col = C["text"] if curve_m == "raw" else C["accent"]
            brush_col = (C["text"] + "20") if curve_m == "raw" else (C["accent"] + "25")

            yy = hist_y(to_plot, norm, log)
            meas_top = float(np.nanmax(yy)) if len(yy) else 1.0
            k = "sum"
            if k in self.rep_curves:
                c = self.rep_curves[k]
                c.setData(xx, yy)
                c.setPen(pg.mkPen(pen_col, width=2.0))
                c.setBrush(pg.mkBrush(brush_col))
                c.setSymbolBrush(pen_col)
                c.opts["name"] = sum_lbl
                c.setVisible(True)
            else:
                c = self.p_rep.plot(
                    xx, yy,
                    pen=pg.mkPen(pen_col, width=2.0),
                    symbol="o", symbolSize=5,
                    symbolPen=None, symbolBrush=pen_col,
                    brush=brush_col, fillLevel=0 if not log else None,
                    name=sum_lbl
                )
                self.rep_curves[k] = c
            active_keys.add(k)

        # Reference trace (HOCAL or IRF) if enabled
        if self.rep_hctl.ref_cb.isChecked() and getattr(self, "rep_hocal_data", None) and f_idx < len(self.rep_hocal_data):
            ref_vec = self.rep_hocal_data[f_idx]
            if ref_vec is not None and len(ref_vec) > 0 and np.nanmax(ref_vec) > 0:
                top = meas_top if meas_top > 0 else 1.0
                sc = (top / np.nanmax(ref_vec)) if np.nanmax(ref_vec) > 0 and top > 0 else 1.0
                rr = ref_vec * sc
                if log:
                    rr = np.clip(rr, top * 1e-5 if norm != "none" else 1.0, None)
                xx_ref = hist_x(len(rr), unit)
                k = "ref"
                if k in self.rep_curves:
                    c = self.rep_curves[k]
                    c.setData(xx_ref, rr)
                    c.setVisible(True)
                else:
                    c = self.p_rep.plot(
                        xx_ref, rr,
                        pen=pg.mkPen(C["muted"], width=1.4, style=Qt.DashLine),
                        symbol="o", symbolSize=3, symbolPen=None, symbolBrush=C["muted"],
                        name="HOCAL"
                    )
                    self.rep_curves[k] = c
                active_keys.add(k)

        # Hide any curves that are not in the active set
        for k in list(self.rep_curves.keys()):
            if k not in active_keys:
                self.rep_curves[k].setData([], [])
                self.rep_curves[k].setVisible(False)

        # Keep legend in sync only when configuration changes (saves massive CPU per frame)
        if hasattr(self.p_rep, "legend") and self.p_rep.legend is not None:
            leg = self.p_rep.legend
            sig = (tuple(sorted(active_keys)), curve_m, self.rep_hctl.legend_cb.isChecked(), sum_lbl if "sum" in active_keys else "")
            if getattr(self, "_last_leg_sig", None) != sig:
                self._last_leg_sig = sig
                leg.clear()
                if not self.rep_hctl.legend_cb.isChecked():
                    leg.hide()
                else:
                    leg.show()
                    cols = 2 if len(active_keys) > 8 else 1
                    leg.setColumnCount(cols)
                    for k in sorted(active_keys):
                        c = self.rep_curves.get(k)
                        if c is not None and c.isVisible() and hasattr(c, "opts") and c.opts.get("name"):
                            leg.addItem(c, c.opts["name"])

        self.p_rep.setTitle(
            f"<span style='color:{C['text']}; font-weight:600;'>pToF Histogram Replay</span> "
            f"<span style='color:{C['muted']}; font-size:9pt;'>[{mode_tag}] · {self.chan.describe()}</span>",
            size="10.5pt"
        )

    # =========================================================================
    # RESULTS & VALIDATION LOGIC
    # =========================================================================
    def _load_results_file(self, path):
        if not path or not os.path.isfile(path):
            return
        self.res_file = path
        companion = None

        # Check if user selected raw recording with companion timeseries
        if path.endswith(".csv") and not (path.endswith("_timeseries.csv") or path.endswith("_live_results.csv") or "results" in path):
            cand1 = path[:-4] + "_timeseries.csv"
            cand2 = path[:-4] + "_live_results.csv"
            ts_dir_cand = os.path.join(os.path.dirname(path), "timeseries", os.path.basename(path)[:-4] + "_timeseries.csv")
            for c in (cand1, cand2, ts_dir_cand):
                if os.path.isfile(c):
                    companion = c
                    break

        load_target = companion if companion else path
        self._parse_results_csv(load_target)

        if companion:
            self.lbl_res_note.setText(f"Loaded companion results: <b>{os.path.basename(companion)}</b>")
        else:
            self.lbl_res_note.setText(f"Loaded: <b>{os.path.basename(path)}</b> ({len(self.res_records)} points)")

        set_pill(self.hdr_pill, f"Results: {len(self.res_records)} pts", C["ok"])
        self.btn_res_csv.setEnabled(True)
        self._populate_table()
        self._render_results_plots()

    def _parse_results_csv(self, path):
        records = []
        try:
            with open(path, "r", errors="ignore") as f:
                header_line = f.readline()
                headers = [h.strip() for h in header_line.split(",")]
                h_map = {h: i for i, h in enumerate(headers)}

                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 5:
                        continue
                    try:
                        rec = {
                            "frame": int(parts[h_map.get("frame", 0)]),
                            "t_s": float(parts[h_map.get("t_s", 1)]),
                            "dT_ps": float(parts[h_map.get("dT_ps", 2)]) if "dT_ps" in h_map else 0.0,
                            "intensity": float(parts[h_map.get("intensity", 3)]) if "intensity" in h_map else 0.0,
                            "mua": float(parts[h_map.get("mua_cm", 4)]) if "mua_cm" in h_map else float(parts[h_map.get("mua", 4)]),
                            "mua_sd": float(parts[h_map.get("mua_sd", 5)]) if "mua_sd" in h_map else 0.005,
                            "musp": float(parts[h_map.get("musp_cm", 6)]) if "musp_cm" in h_map else float(parts[h_map.get("musp", 6)]),
                            "musp_sd": float(parts[h_map.get("musp_sd", 7)]) if "musp_sd" in h_map else 0.25,
                        }
                        records.append(rec)
                    except (ValueError, IndexError):
                        continue
        except Exception as e:
            print("Error parsing results CSV:", e)
        self.res_records = records

    def _load_compare_file(self, path):
        if not path or not os.path.isfile(path):
            return
        try:
            comp = V.read_compare(path)
            if comp:
                # Set expected values from first matching phantom or first row
                first_k = next(iter(comp.keys()))
                ea, es = comp[first_k][:2]
                self.val_mua_exp.setValue(ea)
                self.val_musp_exp.setValue(es)
                self._render_results_plots()
        except Exception as e:
            print("Error loading compare file:", e)

    def _download_compare_template(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Expected Values Template", "compare_template.csv", "CSV files (*.csv)")
        if p:
            with open(p, "w") as f:
                f.write("sample,mua_expected,musp_expected\n")
                f.write("sample_01,0.200,10.00\n")
                f.write("sample_02,0.150,8.50\n")
                f.write("sample_03,0.250,12.00\n")

    def _populate_table(self):
        self.res_table.clear()
        if not self.res_records:
            return
        cols = ["Frame", "Time (s)", "dT (ps)", "Intensity", "μa (cm⁻¹)", "μa SD", "μs′ (cm⁻¹)", "μs′ SD"]
        self.res_table.setColumnCount(len(cols))
        self.res_table.setHorizontalHeaderLabels(cols)
        self.res_table.setRowCount(len(self.res_records))

        for r_idx, r in enumerate(self.res_records):
            vals = [
                str(r["frame"]), f"{r['t_s']:.3f}", f"{r['dT_ps']:.1f}", f"{r['intensity']:.0f}",
                f"{r['mua']:.5f}", f"{r['mua_sd']:.5f}", f"{r['musp']:.4f}", f"{r['musp_sd']:.4f}"
            ]
            for c_idx, v in enumerate(vals):
                it = QtWidgets.QTableWidgetItem(v)
                it.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter if c_idx > 0 else Qt.AlignCenter)
                self.res_table.setItem(r_idx, c_idx, it)

    def _render_results_plots(self):
        if not self.res_records:
            return

        tt = np.array([r["t_s"] for r in self.res_records])
        mua_arr = np.array([r["mua"] for r in self.res_records])
        musp_arr = np.array([r["musp"] for r in self.res_records])

        # Rolling average
        roll_s = self.spin_ts_roll.value()
        if roll_s > 0 and len(tt) > 1:
            dt_s = np.nanmedian(np.diff(tt))
            w = max(1, int(round(roll_s / dt_s))) if dt_s > 0 else 1
            mua_plot = V.moving_average(mua_arr, w) if w > 1 else mua_arr
            musp_plot = V.moving_average(musp_arr, w) if w > 1 else musp_arr
        else:
            mua_plot = mua_arr
            musp_plot = musp_arr

        # Render Time Series (dots and lines)
        roll_tag = f" ({roll_s:g} s moving average)" if (roll_s > 0 and len(tt) > 1 and w > 1) else ""
        self.p_ts_mua.setTitle(f"<span style='color:{C['muted']}; font-size:9.5pt'>Time Series{roll_tag}</span>", justify="left")
        self.c_ts_mua.setData(tt, mua_plot)
        self.c_ts_musp.setData(tt, musp_plot)

        # Update KPIs
        mean_mua = float(np.nanmean(mua_arr))
        sd_mua = float(np.nanstd(mua_arr))
        mean_musp = float(np.nanmean(musp_arr))
        sd_musp = float(np.nanstd(musp_arr))

        self.kpi_mua.set(f"{mean_mua:.4f}", f"± {sd_mua:.4f}")
        self.kpi_musp.set(f"{mean_musp:.3f}", f"± {sd_musp:.3f}")

        # Validation handling
        is_val = self.chk_val_enable.isChecked()
        self.sc_val.setVisible(is_val)
        self.sc_val_status.setVisible(is_val)

        if is_val:
            exp_a = self.val_mua_exp.value()
            exp_s = self.val_musp_exp.value()

            # Time series dashed reference lines
            xx = np.array([tt[0], tt[-1]]) if len(tt) > 1 else np.array([0, 1])
            self.line_exp_mua.setData(xx, [exp_a, exp_a])
            self.line_exp_musp.setData(xx, [exp_s, exp_s])
            self.line_exp_mua.show()
            self.line_exp_musp.show()

            # Discrepancy %
            err_a = ((mean_mua - exp_a) / exp_a) * 100.0 if exp_a > 0 else 0.0
            err_s = ((mean_musp - exp_s) / exp_s) * 100.0 if exp_s > 0 else 0.0

            # 3-tier discrepancy thresholds (≤ 10% green, 10–30% yellow/amber, > 30% red)
            def _tier(err):
                ae = abs(err)
                if ae <= 10.0:
                    return C["ok"], "within ±10%"
                elif ae <= 30.0:
                    return C["warn"], "within ±30%"
                else:
                    return C["bad"], "> ±30%"

            col_a, tier_a_str = _tier(err_a)
            col_s, tier_s_str = _tier(err_s)

            max_err = max(abs(err_a), abs(err_s))
            if max_err <= 10.0:
                pill_col = C["ok"]
                tier_summary = "Within ±10%"
            elif max_err <= 30.0:
                pill_col = C["warn"]
                tier_summary = "Within ±30%"
            else:
                pill_col = C["bad"]
                tier_summary = "> ±30%"

            self.lbl_val_mua.setText(f"Δμa: <b style='color:{col_a}'>{err_a:+.1f} %</b> (Exp: {exp_a:.3f})")
            self.lbl_val_musp.setText(f"Δμs′: <b style='color:{col_s}'>{err_s:+.1f} %</b> (Exp: {exp_s:.2f})")
            set_pill(self.pill_val, f"Δ: {tier_summary} (Δμa {err_a:+.1f}%, Δμs′ {err_s:+.1f}%)", pill_col)
            self.pill_val.setToolTip(
                f"Certified phantom discrepancy:\n"
                f"• μa: recovered {mean_mua:.4f} vs exp {exp_a:.4f} cm⁻¹ (Δ = {err_a:+.1f}%, {tier_a_str})\n"
                f"• μs′: recovered {mean_musp:.3f} vs exp {exp_s:.3f} cm⁻¹ (Δ = {err_s:+.1f}%, {tier_s_str})\n"
                f"Tiers: ≤ 10% (Green) · 10–30% (Amber) · > 30% (Red)"
            )

            # Render 4. Output Validation Plot
            self._draw_val_plots(mean_mua, sd_mua, mean_musp, sd_musp, exp_a, exp_s)
        else:
            self.line_exp_mua.hide()
            self.line_exp_musp.hide()

    def _draw_val_plots(self, mua_m, mua_sd, musp_m, musp_sd, mua_e, musp_e):
        sample_name = os.path.splitext(os.path.basename(self.res_file or "sample"))[0]
        # Build rows compatible with FIG.all_linear
        rows = [(sample_name, mua_m, mua_sd, musp_m, musp_sd, mua_e, musp_e)]
        data = FIG.all_linear(rows, self.chk_ts_err.isChecked())

        for p, which, lab in ((self.p_val_mua, "mua", "μa"), (self.p_val_musp, "musp", "μs′")):
            p.clear()
            lim, fit, pts = data[which]
            xx = np.array([0.0, lim])
            clear = pg.mkPen(0, 0, 0, 0)
            lo = p.plot(xx, 0.9 * xx, pen=clear)
            hi = p.plot(xx, 1.1 * xx, pen=clear)
            band = pg.FillBetweenItem(lo, hi, brush=pg.mkBrush("#dcefe3"))
            band.setZValue(-1)
            p.addItem(band)

            p.plot([0, 0], [0, 0], pen=None, fillLevel=0, brush=pg.mkBrush("#cfe9d8"), name="±10 % band")
            p.plot(xx, xx, pen=pg.mkPen("#777777", width=1.6, style=Qt.DashLine), name="y = x (expected)")

            col = C["mua"] if which == "mua" else C["musp"]
            for name, mk, c, x, y, sd in pts:
                # Dots and lines / points with white border
                p.plot([x], [y], pen=None, symbol="o", symbolSize=11, symbolBrush=col,
                       symbolPen=pg.mkPen("w", width=1.4), name=name)
                if self.chk_ts_err.isChecked():
                    p.addItem(pg.ErrorBarItem(x=np.array([x]), y=np.array([y]), height=np.array([2 * sd]),
                                              pen=pg.mkPen(col, width=1.4), beam=0.08))
                t = pg.TextItem(f"{name}\n({y:.3f})", color="#222222", anchor=(-0.15, 1.0))
                t.setPos(x, y)
                p.addItem(t)

            p.setXRange(0, lim, padding=0.02)
            p.setYRange(0, lim, padding=0.02)

    def _export_results_csv(self):
        p, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export Results Table", "results_export.csv", "CSV files (*.csv)")
        if p and self.res_records:
            with open(p, "w") as f:
                f.write("frame,t_s,dT_ps,intensity,mua_cm,mua_sd,musp_cm,musp_sd\n")
                for r in self.res_records:
                    f.write(f"{r['frame']},{r['t_s']:.3f},{r['dT_ps']:.2f},{r['intensity']:.1f},{r['mua']:.5f},{r['mua_sd']:.5f},{r['musp']:.4f},{r['musp_sd']:.4f}\n")

    def open_in_offline(self):
        target = self.res_file or self.rep_path.text()
        if not target:
            return
        if self.main_win and hasattr(self.main_win, "off"):
            self.main_win.off._add(target)
            self.main_win.bar.setCurrentIndex(2)  # switch to Offline Analysis
            set_pill(self.hdr_pill, f"Transferred to Offline Analysis", C["ok"])
        else:
            print(f"Opening in offline: {target}")

    def shutdown(self):
        self._stop_replay()



