"""
live_engine.py - live acquisition and frame-by-frame moments fitting, using the V12 engine unchanged.

Pieces (no GUI code here, so they can be tested and scripted on their own):
    FrameAssembler   raw sensor lines in -> full, independent frames out (same rules as V12 read_hlong_frames:
                     only genuinely new zone histograms, paired into frames; 4x4 mode updates half the zones per record)
    LiveFitter       frame -> (dT, I) -> mua, musp, exactly like V12 clean_frames + fit_timeseries
    EvmTcpSource     the AMS EVM GUI's TCP port (default 127.0.0.1:39998), as in TMF8828PiDataReader.py
    ReplaySource     plays back a recorded CSV as if it were live (testing, demos, re-analysis)
    Recorder         writes every raw line to a CSV in the EVM log format, so V12 can re-fit it offline
    LiveSession      wires a source -> assembler -> fitter in a background thread; results come out of a queue
"""
import json
import os
import re
import queue
import socket
import threading
import time

import numpy as np
from scipy.optimize import least_squares

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
for p in (ROOT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import tmf_moments_solver as V

TS_TAGS = ("#ITT", "#OBJ", "#TMP", "#LIGHT", "#RES")
HIST_TAGS = ("#HLONG", "#HOCAL0", "#HOCAL1")


# ----------------------------------------------------------------------------- calibration file
def load_calibration(path):
    with open(path) as fh:
        cfg = json.load(fh)
    if not str(cfg.get("format", "")).startswith("ptof-moments-live-calibration"):
        raise ValueError(f"{os.path.basename(path)} is not a live calibration file")
    return cfg


def with_session_dl(cfg, dl_path):
    """the calibration with this session's DL for cleaning the data (V12 --dl vs --ref-dl): t0, K and the IRF stay
    the calibration's (made with its own DL); only the background of the measured histograms uses the new DL."""
    if not dl_path:
        return cfg
    z, n = V.read_hlong(dl_path)
    if not z:
        raise ValueError(f"{os.path.basename(dl_path)} has no histograms")
    c = dict(cfg)
    c["dl"] = {k: [float(x) for x in v] for k, v in z.items()}
    c["dl_pulses"] = n or cfg.get("dl_pulses")
    c["dl_file_session"] = dl_path
    return c


def apply_settings(cfg):
    """Put V12's module settings back exactly as they were when the calibration was made."""
    s = cfg["settings"]
    V.BG_IDX = np.array(s["BG_IDX"], dtype=int)
    V.BG_MODEL = s["BG_MODEL"]; V.PED_START = s["PED_START"]
    V.WIN_PRE = s["WIN_PRE"]; V.WIN_POST = s["WIN_POST"]; V.WIN_SHIFT = s["WIN_SHIFT"]
    V.TREF = s["TREF"]; V.N_TISSUE = s["N_TISSUE"]; V.A_EBC = V._A(V.N_TISSUE)
    V.SIG_T = s["SIG_T"]; V.SIG_I = s["SIG_I"]; V.MIN_SIGNAL = s["MIN_SIGNAL"]
    V.ZONES = set(s["ZONES"]) if s.get("ZONES") else None
    V.MUSP_FIX_FROM = s.get("MUSP_FIX_FROM", "both")
    V.FRAME_PERIOD_S = s.get("FRAME_PERIOD_S", 0.46)


# ----------------------------------------------------------------------------- outside refractive index
# V12 assumes the medium outside the sample is air (n = 1): the boundary term A(n) uses the sample index alone.
# For live fitting the outside index can be set: A then uses the relative index n_sample / n_outside (Contini '97),
# while the speed of light in the sample still uses n_sample. With n_outside = 1 this is exactly V12.
N_EXT = 1.0
_V12_TPSF = V.tpsf
_A_CACHE = {}


def _A_rel(n_rel):
    if abs(n_rel - 1.0) < 1e-6:
        return 1.0                                            # index matched: no reflection at the boundary
    if n_rel not in _A_CACHE:
        _A_CACHE[n_rel] = float(V._A(n_rel))
    return _A_CACHE[n_rel]


def tpsf_ext(rho_mm, mua_cm, musp_cm, n=None):
    n = V.N_TISSUE if n is None else n
    if N_EXT == 1.0:
        return _V12_TPSF(rho_mm, mua_cm, musp_cm, n)
    mua, musp = mua_cm / 10, musp_cm / 10
    A = _A_rel(n / N_EXT)
    v = V.C_MM_PS / n; D = 1 / (3 * musp); z0 = 1 / musp; ze = 2 * A * D
    z3, z4 = -z0, 2 * ze + z0
    t = V.TT
    return -np.exp(-mua*v*t - rho_mm**2/(4*D*v*t)) / (2*(4*np.pi*D*v)**1.5 * t**2.5) * \
           (z3*np.exp(-z3**2/(4*D*v*t)) - z4*np.exp(-z4**2/(4*D*v*t)))


V.tpsf = tpsf_ext          # model_obs() looks tpsf up in the V12 module at call time


def set_outside_index(n_out):
    global N_EXT
    N_EXT = float(n_out)


# ----------------------------------------------------------------------------- finding files a calibration refers to
def _name(p):
    return re.split(r"[\\/]", p or "")[-1]


def resolve_file(stored, rel=None, base_dir=None, search_dirs=()):
    """Find a file a calibration refers to, even if folders were moved / renamed or the path was written on another
    computer: 1) the stored path, 2) the path relative to the calibration file, 3) the same file name in any of
    `search_dirs` (and their 'dl' sub-folder). Returns the path found, or None."""
    if stored and os.path.isfile(stored):
        return stored
    if rel and base_dir:
        p = os.path.normpath(os.path.join(base_dir, rel.replace("\\", os.sep).replace("/", os.sep)))
        if os.path.isfile(p):
            return p
    name = _name(stored or rel or "")
    if name:
        for d in search_dirs:
            for p in (os.path.join(d, name), os.path.join(d, "dl", name), os.path.join(d, "DL", name)):
                if os.path.isfile(p):
                    return p
    return None


def relative_to(path, base_dir):
    try:
        return os.path.relpath(path, base_dir)
    except ValueError:                       # another drive on Windows
        return None


# ----------------------------------------------------------------------------- lines -> frames
class FrameAssembler:
    """Feed raw lines; get full frames back. A frame = one new histogram of every zone (HLONGxx, HOCAL0/1).
    The set of zones is fixed after the first few records (all zones seen by then)."""

    def __init__(self, settle_records=3):
        self.settle = settle_records
        self.n_itt = 0
        self.ts = None
        self.iters = None
        self.last = {}
        self.pending = {}          # zone -> list of (ts, hist)
        self.expected = None
        self.t_first = None
        self.n_out = 0
        self.itt_info = None       # latest #ITT row as a dict (status display)
        self.telemetry = {}        # latest #TMP, #LIGHT, #OBJ, #RES telemetry

    def feed(self, line):
        """Returns a list of frames completed by this line: (t_s_from_start, {zone: hist}, n_pulses)."""
        if line.startswith(TS_TAGS):
            p = V.split_fields(line)
            if len(p) > 1:
                try:
                    self.ts = float(p[1]) / 1000.0
                except ValueError:
                    return []                                  # header row
            if line.startswith("#ITT"):
                if len(p) > 2 and p[2].strip().isdigit():
                    it = int(p[2]) * 1000
                    if self.iters is None:
                        self.iters = it
                    g = lambda i: p[i].strip() if len(p) > i else ""
                    self.itt_info = {"timestamp": g(1), "iterations_k": g(2), "threshold": g(3), "spad_map": g(4),
                                     "range": g(5), "mode": g(6)}
                    self.n_itt += 1
                    if self.expected is None and self.n_itt > self.settle and self.pending:
                        self.expected = set(self.pending) | set(self.last)
                    elif self.expected is not None and self.pending:
                        # pixel layout changed (e.g. 3x3 -> 4x4): zones stopped arriving while others pile up.
                        # Re-learn the zone set from what is arriving now instead of waiting for them forever.
                        starving = [z for z in self.expected if not self.pending.get(z)]
                        if starving and max(len(v) for v in self.pending.values()) > 6:
                            live = {z for z, v in self.pending.items() if v}
                            for z in starving:
                                self.last.pop(z, None)
                            self.expected = live
            elif line.startswith("#TMP") and len(p) > 2:
                self.telemetry["temp"] = p[2].strip()
            elif line.startswith("#LIGHT") and len(p) > 2:
                self.telemetry["ambient"] = p[2].strip()
            elif line.startswith("#OBJ") and len(p) > 3:
                try:
                    z = int(p[2].strip())
                    zones = self.telemetry.setdefault("zones", {})
                    zones[z] = {
                        "d1": p[3].strip() if len(p) > 3 else "",
                        "c1": p[4].strip() if len(p) > 4 else "",
                        "d2": p[5].strip() if len(p) > 5 else "",
                        "c2": p[6].strip() if len(p) > 6 else "",
                    }
                except Exception:
                    pass
            elif line.startswith("#RES") and len(p) > 4:
                try:
                    self.telemetry["photon_obj"] = p[3].strip()
                    self.telemetry["photon_ref"] = p[4].strip()
                except Exception:
                    pass
            return self._emit()
        if line.startswith(HIST_TAGS):
            p = V.split_fields(line)
            if len(p) > 10 and p[1].strip().lstrip("-").isdigit():
                h = np.array([float(x) for x in p[1:] if x.strip() != ""])
                z = p[0]
                if self.expected is not None and z not in self.expected and z.startswith("#HLONG"):
                    # a zone that was not there before (layout changed, e.g. 3x3 -> 4x4): learn the zone set again
                    self.expected = None
                    self.n_itt = 0
                    self.pending = {}
                    self.last = {}
                if z not in self.last or not np.array_equal(h, self.last[z]):
                    self.pending.setdefault(z, []).append((self.ts if self.ts is not None else np.nan, h))
                self.last[z] = h
        return []

    def _emit(self):
        out = []
        if self.expected is None:
            return out
        while all(self.pending.get(z) for z in self.expected):
            items = {z: self.pending[z].pop(0) for z in self.expected}
            tss = np.array([v[0] for v in items.values()], dtype=float)
            t_abs = float(np.nanmean(tss)) if np.isfinite(tss).any() else np.nan
            if self.t_first is None and np.isfinite(t_abs):
                # frames sent before the first timestamp are placed one frame period apart, as V12 does
                self.t_first = t_abs - self.n_out * V.FRAME_PERIOD_S
            t = (t_abs - self.t_first) if (np.isfinite(t_abs) and self.t_first is not None) \
                else self.n_out * V.FRAME_PERIOD_S
            out.append((t, {z: v[1] for z, v in items.items()}, self.iters))
            self.n_out += 1
        return out


# ----------------------------------------------------------------------------- frames -> mua, musp
class LiveFitter:
    """Same maths as V12 clean_frames + fit_timeseries, one frame (or group of frames) at a time.
    Differences from fitting a whole file, by necessity:
      - min-signal blanking compares with the median intensity of the frames so far (not of the whole file)
      - with the 'exp' background the pedestal decay constants come from the first 10 frames (not the file average)
    """

    def __init__(self, cfg, group=1, musp_fix=None, n_index=None, mua_fix=None, fix_from="both", n_outside=1.0):
        """musp_fix / mua_fix (cm-1): hold one property and fit the other (not both). fix_from: which observable(s)
        the remaining property is fitted from - "t" mean time, "I" intensity, "both" the average of the two (as V12
        --fix-musp-from). Fixing mua is not in V12; it mirrors V12's fixed-musp fit with the roles swapped."""
        self.cfg = cfg
        apply_settings(cfg)
        V.MUSP_FIX_FROM = fix_from
        if n_index is not None and n_outside > float(n_index):
            raise ValueError("the outside index must not be larger than the sample index")
        set_outside_index(n_outside)
        self.mua_fix = mua_fix
        self.fix_from = fix_from
        self.rho, self.t0, self.lk = cfg["rho_mm"], cfg["t0_ps"], cfg["logk"]
        self.irf = None if cfg["irf"] is None else np.array(cfg["irf"])
        self.dl = None if cfg["dl"] is None else {k: np.array(v) for k, v in cfg["dl"].items()}
        self.dl_pulses = cfg.get("dl_pulses")
        self.group = max(1, int(group))
        self.musp_fix = musp_fix
        self.n_index = V.N_TISSUE if n_index is None else float(n_index)
        self.buf = []
        self.prev = None
        self.I_hist = []
        self.k = 0
        self.taus = None
        self.exp_backlog = []
        self.warnings = []

    def reset(self):
        """Start again as if at the beginning of a file (used when a recording starts)."""
        self.buf, self.prev, self.I_hist, self.k = [], None, [], 0

    # -- cleaning (V12 clean_frames for one group)
    def _clean(self, frames, n_pulses):
        g = len(frames)
        zones = frames[0][1].keys()
        mean = {z: np.mean([f[1][z] for f in frames], axis=0) for z in zones}
        hk = [z for z in ("#HOCAL0", "#HOCAL1") if z in mean]
        ref = V.clean_hocal(np.mean([mean[z] for z in hk], axis=0)) if hk else None
        y = sum(V.clean_zone(mean[z], None if self.dl is None else self.dl[z], n_pulses,
                             None if self.taus is None else self.taus.get(z))
                for z in mean if (self.dl is None or z in self.dl) and V.use_zone(z))
        return y * g, ref, float(np.mean([f[0] for f in frames])), n_pulses * g

    def add_frame(self, t, zones, n_pulses):
        """Returns a list of results (0 or 1, more when an 'exp' backlog is released)."""
        n_pulses = n_pulses or self.dl_pulses
        if self.dl_pulses and n_pulses and n_pulses != self.dl_pulses and not self.warnings:
            self.warnings.append(f"sensor iterations {n_pulses // 1000}k differ from the DL "
                                 f"({self.dl_pulses // 1000}k) - use a DL recorded with the same iterations")
        if V.BG_MODEL == "exp" and self.taus is None:
            self.exp_backlog.append((t, zones, n_pulses))
            if len(self.exp_backlog) < 10:
                return []
            self.taus = {z: V._fit_ped_taus(V.as_counts(np.mean([f[1][z] for f in self.exp_backlog], axis=0)))
                         for z in zones if z.startswith("#HLONG")}
            out = []
            for f in self.exp_backlog:
                out += self._add(f)
            self.exp_backlog = []
            return out
        return self._add((t, zones, n_pulses))

    def _add(self, frame):
        self.buf.append(frame)
        if len(self.buf) < self.group:
            return []
        frames, self.buf = self.buf, []
        y, ref, t, npul = self._clean(frames, frames[0][2])
        if V.TREF == "hocal":
            if ref is None:
                raise RuntimeError("calibration uses HOCAL but the sensor sends no #HOCAL rows - set histogram "
                                   "mode 3 (reference + measurement)")
            o = V.observables(y, ref, npul)
        else:
            o = V.observables(y, self.irf, npul)
        o = tuple(o) + (self.n_index,)
        res = self._fit(o)
        r = {"frame": self.k, "t_s": t, "fixed": "mua" if self.mua_fix is not None else
             ("musp" if self.musp_fix is not None else ""), "dT_ps": float(o[0]), "intensity": float(o[1]),
             "mua": res[0], "mua_sd": res[1], "musp": res[2], "musp_sd": res[3],
             "hist": y, "ref": ref if ref is not None else self.irf, "window": self._window(ref)}
        self.k += 1
        return [r]

    def _window(self, ref):
        r = ref if ref is not None else self.irf
        k = int(np.argmax(r))
        return (k + V.WIN_SHIFT - V.WIN_PRE, k + V.WIN_SHIFT + V.WIN_POST)

    def _musp_1obs(self, o, k, x0=None):
        """musp from ONE observable with mua fixed (counterpart of V12 _fit_mua_1obs)."""
        rho, t0, lk, mua = self.rho, self.t0, self.lk, self.mua_fix
        F = lambda p: V.residual(mua, p[0], rho, t0, lk, o)[k:k + 1]
        best = None
        for s0 in ((x0,) if x0 is not None else (2.0, 5.0, 10.0, 20.0, 40.0)):
            r = least_squares(F, [s0], bounds=([0.5], [50]))
            if best is None or r.cost < best.cost:
                best = r
        if x0 is not None and (best.cost > 1e-6 or best.x[0] <= 0.51 or best.x[0] >= 49.9):
            return self._musp_1obs(o, k)
        try:
            sd = float(np.sqrt(np.linalg.inv(best.jac.T @ best.jac))[0, 0])
        except np.linalg.LinAlgError:
            sd = np.nan
        return best.x[0], sd

    def _fit_musp_fixed_mua(self, o, x0=None):
        if self.fix_from in ("t", "I"):
            return self._musp_1obs(o, 0 if self.fix_from == "t" else 1, x0)
        mt, st = self._musp_1obs(o, 0, x0)
        mi, si = self._musp_1obs(o, 1, x0)
        return 0.5 * (mt + mi), 0.5 * np.hypot(st, si)

    def _fit(self, o):
        if np.isfinite(o[1]) and o[1] > 0:
            self.I_hist.append(o[1])
        I_ref = float(np.median(self.I_hist)) if self.I_hist else np.nan
        return self.fit_obs(o, I_ref)

    def fit_obs(self, o, I_ref):
        """Fit one observation (dT, I, n). Points with intensity below MIN_SIGNAL x I_ref are left blank (NaN).
        Warm start from the previous point, as V12 fit_timeseries."""
        rho, t0, lk = self.rho, self.t0, self.lk
        nan4 = (np.nan, np.nan, np.nan, np.nan)
        if not (np.isfinite(o[0]) and np.isfinite(o[1]) and o[1] > 0 and o[1] >= V.MIN_SIGNAL * I_ref):
            return nan4
        if self.mua_fix is not None:
            m, sd = self._fit_musp_fixed_mua(o, None if self.prev is None else self.prev[1])
            self.prev = np.array([self.mua_fix, m])
            return float(self.mua_fix), 0.0, float(m), float(sd)
        if self.musp_fix is not None:
            x, err = V.invert_fixed(o, rho, t0, lk, self.musp_fix, None if self.prev is None else self.prev[0])
            self.prev = x
            return float(x[0]), float(err[0]), float(x[1]), float(err[1])
        F = lambda p: V.residual(p[0], p[1], rho, t0, lk, o)
        if self.prev is not None:
            r = least_squares(F, self.prev, bounds=([0.002, 0.5], [3, 50]))
            x, J = r.x, r.jac
            if r.cost > 1.0:
                x, _ = V.invert(o, rho, t0, lk); J = None
        else:
            x, _ = V.invert(o, rho, t0, lk); J = None
        if J is None:
            J = least_squares(F, x, bounds=([0.002, 0.5], [3, 50])).jac
        try:
            err = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
        except np.linalg.LinAlgError:
            err = [np.nan, np.nan]
        self.prev = x
        return float(x[0]), float(err[0]), float(x[1]), float(err[1])


# ----------------------------------------------------------------------------- sources
def _read_all(sock, timeout):
    """Read until the sensor has been quiet for `timeout` s (same approach as TMF8828PiDataReader.read_all)."""
    sock.settimeout(timeout)
    chunks = []
    try:
        while True:
            c = sock.recv(65536)
            if not c:
                break
            chunks.append(c)
    except socket.timeout:
        pass
    return b"".join(chunks).decode("utf-8", errors="ignore")


class EvmTcpSource:
    """Talks to the AMS TMF882x EVM GUI over its TCP port. Each cycle sends the trigger command (default '(m0)',
    one measurement) and passes every line of the reply on."""

    name = "TMF882x sensor"

    def __init__(self, host="127.0.0.1", port=39998, trigger="(m0)", setup=("(H3)",), quiet_s=0.07):
        self.host, self.port, self.trigger, self.setup_cmds, self.quiet = host, int(port), trigger, setup, quiet_s
        self.sock = None
        self._rest = ""

    def open(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=3)
        lines = self._lines(_read_all(self.sock, 0.5))
        for c in self.setup_cmds:
            if c:
                self.sock.sendall(c.encode())
                # a pixel-layout change (P) makes the sensor reconfigure: give it longer before the next command
                time.sleep(0.8 if c.strip().upper().startswith("(P") else 0.1)
                lines += self._lines(_read_all(self.sock, self.quiet))
        return lines

    def send(self, cmd):
        if self.sock:
            self.sock.sendall(cmd.encode())

    def _lines(self, text):
        text = self._rest + text
        parts = text.split("\n")
        self._rest = parts.pop()          # a partial last line waits for the rest
        return [p.rstrip("\r") for p in parts if p.strip()]

    def read(self):
        if self.trigger:
            self.sock.sendall(self.trigger.encode())
        return self._lines(_read_all(self.sock, self.quiet))

    def close(self):
        try:
            if self.sock:
                self.sock.close()
        finally:
            self.sock = None


class ReplaySource:
    """Plays a recorded EVM CSV back record by record. speed = 1 real time, 5 five times faster, 0 as fast as possible."""

    name = "Replay file"

    def __init__(self, path, speed=1.0):
        self.path, self.speed = path, float(speed)
        self.records = None
        self.i = 0
        self.t_prev = None

    def open(self):
        recs, cur, head = [], [], []
        with open(self.path, errors="ignore") as fh:
            for line in fh:
                line = line.rstrip("\r\n")
                if not line.strip():
                    continue
                if line.startswith("#ITT"):
                    p = V.split_fields(line)
                    if len(p) > 1 and p[1].strip().isdigit():
                        if cur:
                            recs.append(cur)
                        cur = []
                cur.append(line)
        if cur:
            recs.append(cur)
        self.records = recs
        return []

    def read(self):
        if self.i >= len(self.records):
            raise EOFError("end of file")
        rec = self.records[self.i]; self.i += 1
        if self.speed > 0:
            t = None
            for l in rec:
                if l.startswith("#ITT"):
                    try:
                        t = float(V.split_fields(l)[1]) / 1000.0
                    except (ValueError, IndexError):
                        pass
            if t is not None and self.t_prev is not None:
                time.sleep(max(0.0, min(2.0, (t - self.t_prev) / self.speed)))
            elif self.t_prev is None:
                pass
            else:
                time.sleep(0.23 / self.speed)
            self.t_prev = t if t is not None else self.t_prev
        return rec

    def progress(self):
        return self.i / max(1, len(self.records or []))

    def send(self, cmd):
        pass

    def close(self):
        pass


class Recorder:
    """Raw lines -> CSV in the EVM log format ('sep=;' first, then the lines as received)."""

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.fh = open(path, "w", newline="")
        self.fh.write("sep=;\n")
        self.n = 0

    def write(self, lines):
        for l in lines:
            self.fh.write(l + "\n")
        self.n += len(lines)
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close(); self.fh = None


class ResultsWriter:
    HEADER = "frame,t_s,dT_ps,intensity,mua_cm,mua_sd,musp_cm,musp_sd,fixed\n"

    def __init__(self, path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self.fh = open(path, "w", newline="")
        self.fh.write(self.HEADER)

    def write(self, r):
        self.fh.write("%d,%.3f,%.2f,%.6g,%.5f,%.5f,%.4f,%.4f,%s\n" % (
            r["frame"], r["t_s"], r["dT_ps"], r["intensity"], r["mua"], r["mua_sd"], r["musp"], r["musp_sd"],
            r.get("fixed", "")))
        self.fh.flush()

    def close(self):
        if self.fh:
            self.fh.close(); self.fh = None


# ----------------------------------------------------------------------------- session
class LiveSession(threading.Thread):
    """Source -> FrameAssembler -> (LiveFitter) in one background thread, with recording switched on and off while it
    runs. Results, frames, status messages and errors are put on self.out as (kind, payload):
        "frame"   {"t_s", "hist", "ref", "n"}       every frame (summed raw histogram, background-free preview)
        "result"  fit result dict                   every fitted point, while live fitting is on
    A recording starts at the next record boundary and restarts frame pairing and the fitter there, so the saved raw
    file re-fitted by V12 gives the same values as the saved live results."""

    def __init__(self, source, fitter=None, dl_path=None):
        super().__init__(daemon=True)
        self.source = source
        self.fitter = fitter
        self.dl_path = dl_path
        self.dl_zones = None
        self.dl_pulses = None
        if dl_path and os.path.isfile(dl_path):
            try:
                self.dl_zones, self.dl_pulses = V.read_dl(dl_path)
            except Exception:
                pass
        self.out = queue.Queue()
        self._stop = threading.Event()
        self.ctl = queue.Queue()
        self.recorder = None
        self.writer = None
        self._pending_rec = None
        self.rec_t0 = None
        self.n_rec_frames = 0

    # -- called from the GUI thread (all go through a queue)
    def stop(self):
        self._stop.set()

    def send_command(self, c):
        self.ctl.put(("cmd", c))

    def set_fitter(self, fitter):
        self.ctl.put(("fitter", fitter))

    def set_dl(self, dl_path):
        self.ctl.put(("dl", dl_path))

    def start_recording(self, raw_path, results_path):
        self.ctl.put(("rec", (raw_path, results_path)))

    def stop_recording(self):
        self.ctl.put(("rec", None))

    # -- worker thread
    def _close_rec(self):
        paths = []
        if self.recorder:
            self.recorder.close(); paths.append(self.recorder.path); self.recorder = None
        if self.writer:
            self.writer.close(); paths.append(self.writer.path); self.writer = None
        self._pending_rec = None
        if paths:
            self.out.put(("rec_stopped", paths))

    def _control(self, asm):
        while not self.ctl.empty():
            what, arg = self.ctl.get()
            if what == "cmd":
                self.source.send(arg)
            elif what == "seq":
                try:
                    count = int(arg)
                    trig = getattr(self.source, "trigger", "(m0)")
                    for _ in range(count):
                        if self._stop.is_set():
                            break
                        self.source.send(trig)
                        time.sleep(0.04)
                except Exception:
                    pass
            elif what == "fitter":
                self.fitter = arg
                if arg is not None and getattr(arg, "dl", None):
                    self.dl_zones = arg.dl
                    self.dl_pulses = getattr(arg, "dl_pulses", None)
                if self.writer and arg is None:
                    self.writer.close(); self.writer = None
                elif arg is not None and self.recorder and not self.writer and self._pending_rec \
                        and self._pending_rec[1]:
                    self.writer = ResultsWriter(self._pending_rec[1])
            elif what == "dl":
                self.dl_path = arg
                if arg and os.path.isfile(arg):
                    try:
                        self.dl_zones, self.dl_pulses = V.read_dl(arg)
                    except Exception:
                        self.dl_zones = None
                else:
                    self.dl_zones = None
            elif what == "rec":
                if arg is None:
                    self._close_rec()
                else:
                    self._pending_rec = arg          # opened at the next #ITT line (record boundary)

    def run(self):
        asm = FrameAssembler()
        shown = set()
        try:
            self.source.open()
            self.out.put(("status", "connected"))
            while not self._stop.is_set():
                self._control(asm)
                try:
                    lines = self.source.read()
                except EOFError:
                    self.out.put(("status", "replay finished")); break
                for line in lines:
                    if self._pending_rec and not self.recorder and line.startswith("#ITT") \
                            and len(V.split_fields(line)) > 1 and V.split_fields(line)[1].strip().isdigit():
                        raw, res = self._pending_rec
                        self.recorder = Recorder(raw)
                        self.n_rec_frames = 0
                        self.writer = ResultsWriter(res) if (self.fitter is not None and res) else None
                        expected = asm.expected
                        asm = FrameAssembler(); asm.expected = expected     # pair frames from the file start, as V12
                        if self.fitter is not None:
                            self.fitter.reset()
                        self.out.put(("rec_started", [p for p in (raw, res if self.writer else None) if p]))
                    if self.recorder:
                        self.recorder.write([line])
                    for t, zones, npul in asm.feed(line):
                        self._frame(t, zones, npul, shown)
                if asm.itt_info:
                    self.out.put(("itt", dict(asm.itt_info)))
                if asm.telemetry:
                    self.out.put(("telemetry", dict(asm.telemetry)))
        except Exception as e:                               # report, never crash the GUI
            self.out.put(("error", f"{type(e).__name__}: {e}"))
        finally:
            try:
                self.source.close()
            finally:
                self._close_rec()
                self.out.put(("stopped", None))

    def _frame(self, t, zones, npul, shown):
        hk = [k for k in ("#HOCAL0", "#HOCAL1") if k in zones]
        ref = V.clean_hocal(np.mean([zones[k] for k in hk], axis=0)) if hk else None

        # 1. Raw channels & sum
        chans_raw = {int(k[6:]): np.asarray(v, float) for k, v in zones.items() if k.startswith("#HLONG")}
        # 2. Baseline zeroed (floor subtracted)
        chans_floor = {ch: v - np.median(v[:12]) for ch, v in chans_raw.items()}

        # 3. Cleaned (DL-subtracted) & Background model
        chans_clean = {}
        chans_bg = {}
        dl_dict = self.fitter.dl if (self.fitter and getattr(self.fitter, "dl", None)) else self.dl_zones
        taus = self.fitter.taus if self.fitter else None
        for k, v in zones.items():
            if not k.startswith("#HLONG"):
                continue
            ch = int(k[6:])
            dl_z = dl_dict.get(k) if dl_dict else None
            try:
                c_z = V.clean_zone(v, dl_z, npul, taus.get(k) if taus else None)
                chans_clean[ch] = c_z
                chans_bg[ch] = np.asarray(v, float) - c_z
            except Exception:
                chans_clean[ch] = chans_floor.get(ch, np.asarray(v, float))
                chans_bg[ch] = np.zeros_like(v, dtype=float)

        n_bins = len(next(iter(chans_raw.values()))) if chans_raw else 128
        y_raw = sum(chans_raw.values()) if chans_raw else np.zeros(n_bins)
        y_floor = sum(chans_floor.values()) if chans_floor else np.zeros(n_bins)
        y_clean = sum(chans_clean.values()) if chans_clean else np.zeros(n_bins)
        y_bg = sum(chans_bg.values()) if chans_bg else np.zeros(n_bins)

        self.n_rec_frames += 1
        self.out.put(("frame", {
            "t_s": t,
            "hist": y_floor,            # default baseline-zeroed for backward compatibility
            "hist_raw": y_raw,
            "hist_floor": y_floor,
            "hist_clean": y_clean,
            "hist_bg": y_bg,
            "ref": ref,
            "recording": bool(self.recorder),
            "counts": float(y_floor[12:].sum()) if len(y_floor) > 12 else float(y_floor.sum()),
            "channels": chans_floor,    # default for backward compatibility
            "channels_raw": chans_raw,
            "channels_floor": chans_floor,
            "channels_clean": chans_clean,
            "channels_bg": chans_bg,
            "has_dl": bool(dl_dict),
        }))
        if self.fitter is None:
            return
        for r in self.fitter.add_frame(t, zones, npul):
            r["frame"] = self.n_rec_frames - 1          # frame number in the recording (fitter may restart)
            if self.writer:
                self.writer.write(r)
            self.out.put(("result", r))
        for w in self.fitter.warnings:
            if w not in shown:
                shown.add(w); self.out.put(("warning", w))
