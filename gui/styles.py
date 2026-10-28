"""UI Styling, themes, palettes, and visual helper builders."""
import os
import sys
import warnings
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
import pyqtgraph as pg

# Configure QCheckBox so that indicator appears after words (text first, box after)
_orig_qcheckbox_init = QtWidgets.QCheckBox.__init__


def _custom_qcheckbox_init(self, *args, **kwargs):
    _orig_qcheckbox_init(self, *args, **kwargs)
    self.setLayoutDirection(QtCore.Qt.RightToLeft)
    self.setSizePolicy(QtWidgets.QSizePolicy.Maximum, QtWidgets.QSizePolicy.Fixed)


QtWidgets.QCheckBox.__init__ = _custom_qcheckbox_init

# In vertical box layouts (e.g. cards), ensure RTL checkboxes anchor to the left margin
# instead of stretching across to the right edge.
_orig_box_add_widget = QtWidgets.QBoxLayout.addWidget


def _custom_box_add_widget(self, w, stretch=0, alignment=QtCore.Qt.Alignment()):
    if isinstance(w, QtWidgets.QCheckBox) and int(alignment) == 0:
        if self.direction() in (QtWidgets.QBoxLayout.TopToBottom, QtWidgets.QBoxLayout.BottomToTop):
            alignment = QtCore.Qt.AlignLeft | QtCore.Qt.AlignAbsolute
    return _orig_box_add_widget(self, w, stretch, alignment)


QtWidgets.QBoxLayout.addWidget = _custom_box_add_widget

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(GUI_DIR)
CORE_DIR = os.path.join(ROOT_DIR, "core")
for p in (ROOT_DIR, CORE_DIR, GUI_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

HERE = ROOT_DIR
APP_NAME = "tmf momenTOF"
ENGINE = os.path.join(ROOT_DIR, "tmf_moments_solver.py")
CAL_CLI = os.path.join(CORE_DIR, "calibrate_cli.py") if os.path.isfile(os.path.join(CORE_DIR, "calibrate_cli.py")) else os.path.join(ROOT_DIR, "calibrate_cli.py")
OFFLINE = os.path.join(CORE_DIR, "offline_engine.py") if os.path.isfile(os.path.join(CORE_DIR, "offline_engine.py")) else os.path.join(ROOT_DIR, "offline_engine.py")

_PG_SYM = {"o": "o", "s": "s", "^": "t1", "D": "d", "v": "t", "P": "+", "X": "x", "*": "star",
           "h": "h", "p": "p"}

# ----------------------------------------------------------------------------- look & theme
C = dict(
    bg="#0d1016",
    panel="#131820",
    card="#181e29",
    card2="#202836",
    border="#2a3344",
    text="#edf0f5",
    muted="#8c96a9",
    accent="#ec5f9f",
    accent_soft="#3b1d30",
    mua="#4f8cff",
    musp="#ff9f43",
    inten="#3ecf8e",
    dt="#b48cff",
    ok="#3ecf8e",
    warn="#f5b14c",
    bad="#ff5d6c",
)

TICK_SVG = os.path.join(GUI_DIR, "tick_pink.svg").replace("\\", "/")
TICK_DISABLED_SVG = os.path.join(GUI_DIR, "tick_muted.svg").replace("\\", "/")


def _ensure_tick_icons():
    svg_pink = f'''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
  <polyline points="3.5,8.5 6.5,11.5 12.5,4.5" fill="none" stroke="{C['accent']}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    svg_muted = f'''<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
  <polyline points="3.5,8.5 6.5,11.5 12.5,4.5" fill="none" stroke="{C['muted']}" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round"/>
</svg>'''
    try:
        if not os.path.isfile(TICK_SVG):
            with open(TICK_SVG, "w", encoding="utf-8") as f:
                f.write(svg_pink)
        if not os.path.isfile(TICK_DISABLED_SVG):
            with open(TICK_DISABLED_SVG, "w", encoding="utf-8") as f:
                f.write(svg_muted)
    except Exception:
        pass


_ensure_tick_icons()

QSS = f"""
* {{
    font-family: "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 9.5pt;
    color: {C['text']};
}}
QMainWindow, QWidget#root {{
    background: {C['bg']};
}}
QTabWidget::pane {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    background: {C['card']};
    top: -1px;
}}
QTabBar {{
    background: transparent;
}}
QTabBar::tab {{
    background: transparent;
    color: {C['muted']};
    padding: 10px 20px;
    margin-right: 6px;
    border: none;
    border-bottom: 2px solid transparent;
    font-weight: 600;
    font-size: 10pt;
}}
QTabBar::tab:selected {{
    color: {C['text']};
    border-bottom: 2px solid {C['accent']};
}}
QTabBar::tab:hover {{
    color: {C['text']};
}}
QTabWidget#calTabs::pane {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    background: {C['panel']};
    top: -1px;
    padding: 6px;
}}
QTabWidget#calTabs QTabBar::tab {{
    padding: 6px 14px;
    font-size: 9pt;
}}
QFrame#card {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
}}
QFrame#subcard {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 8px;
}}
QFrame#kpi {{
    background: {C['card']};
    border: 1px solid {C['border']};
    border-radius: 10px;
}}
QLabel#cardTitle {{
    color: {C['accent']};
    font-size: 8.5pt;
    font-weight: 700;
    letter-spacing: 0.8px;
}}
QLabel#sectionTitle {{
    color: {C['text']};
    font-size: 9.5pt;
    font-weight: 600;
}}
QLabel#muted {{
    color: {C['muted']};
    font-size: 9pt;
}}
QLabel#kpiValue {{
    font-size: 19pt;
    font-weight: 600;
}}
QLabel#kpiTitle {{
    color: {C['muted']};
    font-size: 8pt;
    font-weight: 700;
    letter-spacing: 0.6px;
}}
QLabel#title {{
    font-size: 14pt;
    font-weight: 700;
}}
QLabel#pill {{
    border-radius: 11px;
    padding: 4px 12px;
    font-weight: 600;
    font-size: 8.5pt;
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {C['accent']};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {C['accent']};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: {C['muted']};
    background: #10141c;
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background: {C['card2']};
    border: 1px solid {C['border']};
    selection-background-color: {C['accent_soft']};
}}
QPushButton {{
    background: {C['card2']};
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 6px 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    border-color: {C['accent']};
    background: #263042;
}}
QPushButton:disabled {{
    color: {C['muted']};
    background: {C['panel']};
    border-color: #1f2633;
}}
QPushButton#primary {{
    background: {C['accent']};
    border: 1px solid {C['accent']};
    color: #ffffff;
    padding: 8px 16px;
    font-weight: 600;
}}
QPushButton#primary:hover {{
    background: #f278b0;
    border-color: #f278b0;
}}
QPushButton#primary:disabled {{
    background: #422536;
    border-color: #422536;
    color: #a88397;
}}
QPushButton#ghost {{
    background: transparent;
    border: 1px solid {C['border']};
    color: {C['accent']};
    padding: 4px 10px;
    font-size: 8.5pt;
    font-weight: 600;
}}
QPushButton#ghost:hover {{
    border-color: {C['accent']};
    background: {C['accent_soft']};
}}
QPushButton#danger {{
    background: #351c24;
    border: 1px solid #6b2a36;
    color: #ffb3bb;
    padding: 8px 14px;
}}
QPushButton#danger:hover {{
    border-color: {C['bad']};
    background: #47222d;
}}
QPushButton#danger:disabled {{
    background: {C['panel']};
    color: {C['muted']};
    border-color: {C['border']};
}}
QPushButton#record {{
    background: #2f1923;
    border: 1px solid #6e2b42;
    color: #ff9eb5;
    padding: 8px 16px;
}}
QPushButton#record:hover {{
    border-color: {C['accent']};
    background: #3d202e;
}}
QPushButton#record[rec="true"] {{
    background: {C['bad']};
    color: white;
    border: 1px solid {C['bad']};
}}
QPushButton#record:disabled {{
    background: {C['panel']};
    color: {C['muted']};
    border-color: {C['border']};
}}
QPushButton#chan {{
    padding: 2px;
    font-weight: 600;
    background: {C['panel']};
    color: {C['muted']};
    border-radius: 5px;
}}
QPushButton#chan:checked {{
    background: {C['accent_soft']};
    color: {C['text']};
    border-color: {C['accent']};
}}
QPushButton#chan[fit="true"] {{
    border: 2px solid {C['ok']};
}}
QPushButton#chan:disabled {{
    background: transparent;
    color: {C['border']};
    border: 1px dashed {C['border']};
}}
QPushButton#modeBtn {{
    padding: 7px 12px;
    font-weight: 600;
    font-size: 8.5pt;
    background: {C['panel']};
    color: {C['muted']};
    border: 1px solid {C['border']};
    border-radius: 6px;
}}
QPushButton#modeBtn:hover {{
    color: {C['text']};
    border-color: {C['accent']};
    background: {C['card2']};
}}
QPushButton#modeBtn:checked {{
    background: {C['accent']};
    color: #ffffff;
    border: 1px solid {C['accent']};
}}
QPushButton#modeBtn:checked:hover {{
    background: #f278b0;
    border-color: #f278b0;
}}
QPushButton#small {{
    padding: 4px 10px;
    font-weight: 500;
    font-size: 9pt;
}}
QCheckBox {{
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid {C['border']};
    background: {C['panel']};
}}
QCheckBox::indicator:hover {{
    border-color: {C['accent']};
}}
QCheckBox::indicator:pressed {{
    border-color: {C['accent']};
    background: {C['accent_soft']};
}}
QCheckBox::indicator:checked {{
    background: {C['panel']};
    border: 1px solid {C['accent']};
    image: url({TICK_SVG});
}}
QCheckBox::indicator:checked:hover {{
    background: {C['accent_soft']};
    border-color: {C['accent']};
    image: url({TICK_SVG});
}}
QCheckBox::indicator:checked:disabled {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    image: url({TICK_DISABLED_SVG});
}}
QRadioButton {{
    spacing: 8px;
}}
QRadioButton::indicator {{
    width: 15px;
    height: 15px;
    border-radius: 8px;
    border: 1px solid {C['border']};
    background: {C['panel']};
}}
QRadioButton::indicator:checked {{
    background: {C['accent']};
    border: 3px solid {C['panel']};
}}
QScrollArea {{
    border: none;
    background: transparent;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {C['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {C['muted']};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {C['border']};
    border-radius: 4px;
    min-width: 30px;
}}
QPlainTextEdit#log {{
    font-family: "Cascadia Mono", "Consolas", "DejaVu Sans Mono", monospace;
    font-size: 9pt;
    background: #0a0d12;
    border: 1px solid {C['border']};
    border-radius: 6px;
}}
QTableWidget {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    gridline-color: {C['border']};
    alternate-background-color: #161c26;
}}
QHeaderView {{
    background: {C['card2']};
    border: none;
}}
QTableCornerButton::section {{
    background: {C['card2']};
    border: none;
}}
QHeaderView::section {{
    background: {C['card2']};
    color: {C['muted']};
    border: none;
    border-bottom: 1px solid {C['border']};
    padding: 7px 8px;
    font-weight: 700;
    font-size: 8.5pt;
}}
QListWidget {{
    background: {C['panel']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 3px;
}}
QListWidget::item {{
    padding: 5px 8px;
    border-radius: 4px;
}}
QListWidget::item:selected {{
    background: {C['accent_soft']};
    color: {C['text']};
}}
QToolButton#section {{
    border: none;
    color: {C['muted']};
    font-weight: 600;
    font-size: 8.5pt;
    padding: 4px 0;
}}
QToolButton#section:hover {{
    color: {C['text']};
}}
QSplitter::handle {{
    background: {C['bg']};
    width: 8px;
    height: 8px;
}}
QToolTip {{
    background: {C['card2']};
    color: {C['text']};
    border: 1px solid {C['border']};
    padding: 6px 8px;
    border-radius: 4px;
}}
"""

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN plots before the first fit
pg.setConfigOptions(antialias=True, background=C["card"], foreground=C["muted"])


def card(title=None, subtitle=None):
    f = QtWidgets.QFrame()
    f.setObjectName("card")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(14, 12, 14, 14)
    lay.setSpacing(8)
    if title:
        t = QtWidgets.QLabel(title.upper())
        t.setObjectName("cardTitle")
        lay.addWidget(t)
    if subtitle:
        s = muted(subtitle)
        lay.addWidget(s)
    return f, lay


def subcard():
    f = QtWidgets.QFrame()
    f.setObjectName("subcard")
    lay = QtWidgets.QVBoxLayout(f)
    lay.setContentsMargins(10, 8, 10, 8)
    lay.setSpacing(4)
    return f, lay


def muted(text):
    l = QtWidgets.QLabel(text)
    l.setObjectName("muted")
    l.setWordWrap(True)
    return l


def rgba(color, alpha):
    """'#rrggbb' + alpha 0-255 -> 'rgba(r,g,b,a)' (Qt style sheets read #rrggbbaa as #aarrggbb)."""
    c = QtGui.QColor(color)
    return f"rgba({c.red()},{c.green()},{c.blue()},{alpha})"


def pill(text, color):
    l = QtWidgets.QLabel(text)
    l.setObjectName("pill")
    set_pill(l, text, color)
    return l


def set_pill(l, text, color):
    l.setText(text)
    if l.objectName() == "connPill":
        l.setStyleSheet(
            f"background: {rgba(color, 36)}; color: {color}; "
            f"border: 1.5px solid {rgba(color, 140)}; "
            f"padding: 6px 18px; border-radius: 14px; "
            f"font-size: 11pt; font-weight: 700;"
        )
    else:
        l.setStyleSheet(f"background: {rgba(color, 36)}; color: {color}; border: 1px solid {rgba(color, 110)};")
    if not text:
        l.setVisible(False)


class CountAxisItem(pg.AxisItem):
    """Clean, human-readable axis tick formatter for photon counts and intensities.
    In log mode: displays powers of 10 clearly: 1, 10, 100, 1k, 10k, 100k, 1M.
    In linear mode: displays clean SI numbers: 0, 20k, 40k, 60k, 80k, 100k."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm_mode = "none"

    def tickStrings(self, values, scale, spacing):
        if self.norm_mode != "none":
            return super().tickStrings(values, scale, spacing)
        strings = []
        is_log = getattr(self, "logMode", False)
        for v in values:
            if is_log:
                try:
                    actual = 10.0 ** v
                except Exception:
                    actual = v
            else:
                actual = v * scale
            sign = "-" if actual < -1e-9 else ""
            mag = abs(actual)
            if mag < 1e-9:
                strings.append("0")
            elif mag >= 1e6:
                strings.append(f"{sign}{mag/1e6:.2g}M")
            elif mag >= 1e3:
                strings.append(f"{sign}{mag/1e3:.3g}k")
            elif mag >= 1:
                strings.append(f"{sign}{mag:.0f}")
            else:
                strings.append(f"{sign}{mag:.2g}")
        return strings


def style_plot(p, left, bottom=None, units_left=None):
    p.showGrid(x=True, y=True, alpha=0.18)
    p.setLabel("left", left, units=units_left)
    if bottom:
        p.setLabel("bottom", bottom)
    for ax in ("left", "bottom"):
        a = p.getAxis(ax)
        a.setPen(pg.mkPen(C["border"]))
        a.setTextPen(pg.mkPen(C["muted"]))
    p.getAxis("left").enableAutoSIPrefix(False)
    p.getViewBox().setBackgroundColor(C["card"])
    return p




def _icon(draw, size=18):
    """Small line icon drawn in the text colour (crisp on high-DPI screens)."""
    scale = 3
    pm = QtGui.QPixmap(size * scale, size * scale)
    pm.fill(Qt.transparent)
    p = QtGui.QPainter(pm)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.scale(scale, scale)
    pen = QtGui.QPen(QtGui.QColor(C["text"]), 1.5)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    p.setPen(pen)
    draw(p, pen)
    p.end()
    pm.setDevicePixelRatio(scale)
    return QtGui.QIcon(pm)


def _draw_limits(p, pen):        # axes with arrowheads and a dashed range box
    P = QtCore.QPointF
    p.drawPolyline([P(3, 2.5), P(3, 15), P(15.5, 15)])
    p.drawPolyline([P(1.5, 4.2), P(3, 2.5), P(4.5, 4.2)])
    p.drawPolyline([P(13.8, 13.5), P(15.5, 15), P(13.8, 16.5)])
    pen.setStyle(Qt.DashLine)
    pen.setWidthF(1.2)
    p.setPen(pen)
    p.drawRect(QtCore.QRectF(6.5, 5.5, 6.5, 6.5))


def _draw_camera(p, _pen):
    path = QtGui.QPainterPath()
    path.addRoundedRect(QtCore.QRectF(1.8, 5, 14.4, 10), 2, 2)
    p.drawPath(path)
    p.drawPolyline([QtCore.QPointF(6, 5), QtCore.QPointF(7.3, 2.8), QtCore.QPointF(10.7, 2.8), QtCore.QPointF(12, 5)])
    p.drawEllipse(QtCore.QPointF(9, 10), 3, 3)


def _draw_html(p, _pen):          # </> : interactive web page
    P = QtCore.QPointF
    p.drawPolyline([P(6, 4.5), P(1.8, 9), P(6, 13.5)])
    p.drawPolyline([P(12, 4.5), P(16.2, 9), P(12, 13.5)])
    p.drawLine(P(10.4, 3), P(7.6, 15))


def icon_button(draw, tip, label=None):
    b = QtWidgets.QPushButton(label or "")
    b.setObjectName("small")
    b.setIcon(_icon(draw))
    b.setIconSize(QtCore.QSize(16, 16))
    b.setToolTip(tip)
    if not label:
        b.setFixedWidth(34)
    b.setCursor(Qt.PointingHandCursor)
    return b



CHANNEL_COLORS = [
    "#00d4ff",  # Cyan (Zone 0)
    "#ff477e",  # Magenta/Pink (Zone 1)
    "#2bd576",  # Green (Zone 2)
    "#ffb830",  # Amber/Orange (Zone 3)
    "#a78bfa",  # Purple (Zone 4)
    "#38bdf8",  # Sky (Zone 5)
    "#f43f5e",  # Rose (Zone 6)
    "#10b981",  # Emerald (Zone 7)
    "#f59e0b",  # Yellow-orange (Zone 8)
    "#e879f9",  # Fuchsia
    "#60a5fa",  # Light blue
    "#34d399",  # Mint
    "#fbbf24",  # Warm amber
    "#f87171",  # Light red
    "#818cf8",  # Indigo
    "#c084fc",  # Violet
]


