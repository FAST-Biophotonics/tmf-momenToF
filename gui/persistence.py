"""State persistence helpers (QSettings, configuration export/import)."""
import os
import json
import datetime as dt
from PySide6 import QtCore, QtWidgets
from gui.styles import APP_NAME

SETTINGS_FORMAT = "ptof-moments-settings/1"

def widget_values(widgets):
    from gui.widgets import PathEdit
    out = {}
    for k, w in widgets.items():
        if isinstance(w, (PathEdit, QtWidgets.QLineEdit)):
            out[k] = w.text()
        elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
            out[k] = w.value()
        elif isinstance(w, QtWidgets.QComboBox):
            out[k] = w.currentText()
        elif isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
            out[k] = bool(w.isChecked())
    return out

def set_widget_values(widgets, values):
    from gui.widgets import PathEdit
    if "irf" in values and "tz_irf" not in widgets and str(values.get("tz_hocal", "")).lower() in ("true", "1"):
        values = dict(values, irf="")
    for k, w in widgets.items():
        if k not in values:
            continue
        v = values[k]
        try:
            if isinstance(w, (PathEdit, QtWidgets.QLineEdit)):
                w.setText(str(v))
            elif isinstance(w, QtWidgets.QSpinBox):
                w.setValue(int(float(v)))
            elif isinstance(w, QtWidgets.QDoubleSpinBox):
                w.setValue(float(v))
            elif isinstance(w, QtWidgets.QComboBox):
                i = w.findText(str(v))
                if i >= 0:
                    w.setCurrentIndex(i)
                elif w.isEditable():
                    w.setEditText(str(v))
            elif isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                w.setChecked(v if isinstance(v, bool) else str(v).lower() in ("true", "1"))
        except (TypeError, ValueError):
            pass

def read_app_json(path):
    """Any app JSON -> (kind, form inputs, full dict). kind: 'calibration' or 'settings'."""
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    fmt = str(d.get("format", ""))
    if fmt.startswith("ptof-moments-live-calibration"):
        return "calibration", d.get("gui_inputs") or {}, d
    if fmt.startswith("ptof-moments-settings"):
        return "settings", d.get("inputs") or {}, d
    raise ValueError(f"{os.path.basename(path)} is not a {APP_NAME} calibration or settings file")

def write_settings(path, tab, widgets):
    d = {"format": SETTINGS_FORMAT, "tab": tab, "created": dt.datetime.now().isoformat(timespec="seconds"),
         "inputs": widget_values(widgets)}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(d, fh, indent=1, ensure_ascii=False)

class Persist:
    """Save / restore every named input widget under a prefix with QSettings."""

    def __init__(self):
        self.s = QtCore.QSettings("FAST-Biophotonics", "pToF-Moments")

    def save(self, prefix, widgets):
        from gui.widgets import PathEdit
        for k, w in widgets.items():
            key = f"{prefix}/{k}"
            if isinstance(w, (PathEdit, QtWidgets.QLineEdit)):
                self.s.setValue(key, w.text())
            elif isinstance(w, (QtWidgets.QSpinBox, QtWidgets.QDoubleSpinBox)):
                self.s.setValue(key, w.value())
            elif isinstance(w, QtWidgets.QComboBox):
                self.s.setValue(key, w.currentText())
            elif isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                self.s.setValue(key, bool(w.isChecked()))

    def load(self, prefix, widgets):
        from gui.widgets import PathEdit
        for k, w in widgets.items():
            key = f"{prefix}/{k}"
            if not self.s.contains(key):
                continue
            v = self.s.value(key)
            try:
                if isinstance(w, (PathEdit, QtWidgets.QLineEdit)):
                    w.setText(str(v))
                elif isinstance(w, QtWidgets.QSpinBox):
                    w.setValue(int(float(v)))
                elif isinstance(w, QtWidgets.QDoubleSpinBox):
                    w.setValue(float(v))
                elif isinstance(w, QtWidgets.QComboBox):
                    i = w.findText(str(v))
                    if i >= 0:
                        w.setCurrentIndex(i)
                elif isinstance(w, (QtWidgets.QCheckBox, QtWidgets.QRadioButton)):
                    w.setChecked(str(v).lower() in ("true", "1"))
            except (TypeError, ValueError):
                pass
