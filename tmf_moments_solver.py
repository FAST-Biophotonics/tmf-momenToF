"""
tmf_moments_solver.py - Analytical Contini '97 moments analysis solver for TMF882x pToF data.
Calculates mean time and intensity moments to recover optical properties (absorption μa, reduced scattering μs').
==========================================================================================
"""
import argparse, json, os, re, sys, time
import numpy as np
from scipy.optimize import least_squares

DT_PS = 381.0 #resolution of bins
C_MM_PS = 0.299792458 #speed of light @ n = 1 [in mm per ps]


# ----------------------------------------------------------------------------- reading
_SPLIT = re.compile(r"[;,\t]")


def split_fields(line):
    """Split a log line on ';', ',' or tab, so files re-saved by Excel with commas read the same as the originals."""
    return _SPLIT.split(line.strip())


def read_hlong(path): #reading our files
    """Return ({zone: frame-averaged histogram}, iterations_in_pulses)."""
    rows, iters = {}, None
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("#HLONG"): #consider all lines beginning with HLONG
                p = split_fields(line)
                if len(p) > 10 and p[1].strip().lstrip("-").isdigit(): #check line is longer than 10 to prove it is data

                #converts the counts to numbers and files them by pixel
                    rows.setdefault(p[0], []).append([float(x) for x in p[1:] if x.strip() != ""])

            elif line.startswith("#ITT") and iters is None: #Lines starting with #ITT hold the acquisition settings
                p = split_fields(line)
                if len(p) > 2 and p[2].strip().isdigit(): #p[2] is no. iterations (laser pulses per frame)
                    iters = int(p[2]) * 1000          # #ITT reports kilo-iterations so *1000 to convert to pulses
    return {k: np.mean(v, axis=0) for k, v in rows.items()}, iters #average frames by bin
   #average counts leaves here

def read_hlong_iters(path):
    """Iteration count (pulses) of a file from its first #ITT line."""
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("#ITT"):
                p = split_fields(line)
                if len(p) > 2 and p[2].strip().isdigit():
                    return int(p[2]) * 1000
    return None


def start_time(path):
    """Time the recording started (s since 1970), from the first #ITT timestamp. None if unusable."""
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith("#ITT"):
                p = split_fields(line)
                if len(p) > 1:
                    try:
                        v = float(p[1])
                        if v > 1e11 and v == int(v):         # ms timestamp (Excel re-saves give 1.79E+12 -> lost)
                            return v / 1000.0
                    except ValueError:
                        pass
    return None


TREF = "irf"     # time reference: "irf" (separate IRF file) or "hocal" (--hocal: the file's own reference histogram)


def clean_hocal(h):
    """Reference histogram minus its pre-pulse floor (bins 0-11). The floor is ~25 counts against a ~260k peak."""
    h = np.asarray(h, dtype=float)
    return h - np.median(h[:12])


def read_hocal(path):
    """Frame-averaged reference histogram of a file: mean of #HOCAL0 and #HOCAL1 (the two halves of a 4x4 frame)."""
    rows = []
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith(("#HOCAL0", "#HOCAL1")):
                p = split_fields(line)
                if len(p) > 10 and p[1].strip().lstrip("-").isdigit():
                    rows.append([float(x) for x in p[1:] if x.strip() != ""])
    if not rows:
        raise SystemExit(f"--hocal: no #HOCAL rows in {os.path.basename(path)} - this file has no reference histogram "
                         f"(record with histogram mode 'reference + measurement'), or run with an IRF instead")
    return clean_hocal(np.mean(rows, axis=0))


def time_ref(path, irf):
    """What the mean arrival time is measured against: the IRF histogram, or (--hocal) the file's own HOCAL."""
    return read_hocal(path) if TREF == "hocal" else irf


# -----------------------------------------------------------------------------
BG_IDX = np.r_[0:12, 70:128] #clean bins (signal-free); change with --bg-bins
BG_MODEL = "dl"      # "dl": a + b*DL pedestal (needs --dl);  "exp": a + c*exp(-(k-K0)/tau) after the crosstalk;  "flat": a
PED_START = 13       # first bin of the crosstalk/pedestal (bins before it are pure pre-pulse floor), used by "exp"


def parse_bins(text):
    """'0-12,18-37,76-127' -> array of bin indices (ranges inclusive)."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-"); out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return np.array(sorted(set(out)))


def _fit_ped_taus(r):
    """Find the two decay constants (in bins) of the crosstalk pedestal from the background bins after PED_START."""
    post = BG_IDX[BG_IDX >= PED_START]
    best, taus = None, np.geomspace(2, 1000, 40)
    for i, t1 in enumerate(taus):
        for t2 in taus[i + 1:]:
            X = np.c_[np.ones(len(post)), np.exp(-(post - PED_START) / t1), np.exp(-(post - PED_START) / t2)]
            c, *_ = np.linalg.lstsq(X, r[post], rcond=None)
            err = np.sum((X @ c - r[post]) ** 2)
            if best is None or err < best[0]:
                best = (err, t1, t2)
    return best[1], best[2]


def _exp_pedestal(r, taus=None):
    """Background without a DL file: a flat floor before PED_START (mean of the background bins there), and from
    PED_START on  a + c1*exp(-(k-PED_START)/tau1) + c2*exp(-(k-PED_START)/tau2)  (fast crosstalk decay + slow pedestal),
    fitted to the background bins at or after PED_START. taus can be passed in (fitted once on the file average)."""
    k = np.arange(len(r))
    pre = BG_IDX[BG_IDX < PED_START]
    post = BG_IDX[BG_IDX >= PED_START]
    bg = np.full(len(r), r[pre].mean() if len(pre) else 0.0)
    if len(post) < 5:
        return bg
    t1, t2 = taus if taus is not None else _fit_ped_taus(r)
    X = np.c_[np.ones(len(post)), np.exp(-(post - PED_START) / t1), np.exp(-(post - PED_START) / t2)]
    c, *_ = np.linalg.lstsq(X, r[post], rcond=None)
    m = k >= PED_START
    bg[m] = c[0] + c[1] * np.exp(-(k[m] - PED_START) / t1) + c[2] * np.exp(-(k[m] - PED_START) / t2)
    return bg


ZONES = None         # --zones: set of '#HLONGxx' names to use (None = all 16)


def use_zone(k):
    return k.startswith("#HLONG") and (ZONES is None or k in ZONES)


def parse_zones(text):
    """'6,7,10,11' -> {'#HLONG06', '#HLONG07', '#HLONG10', '#HLONG11'}"""
    return {f"#HLONG{int(z):02d}" for z in text.split(",") if z.strip()}


def as_counts(h):
    """Frame-averaged counts as floats. (No pile-up correction in V6: tests showed it changes results by < 1 %.)"""
    return np.asarray(h, dtype=float)


#preprocessing step
def clean_zone(raw, dl, n_pulses, taus=None):
    r = as_counts(raw)
    if BG_MODEL == "flat":
        return r - r[BG_IDX].mean()
    if BG_MODEL == "exp":
        return r - _exp_pedestal(r, taus)
    d = as_counts(dl)

    #takes average of the DL's first 12 bins (before the pulse) and subtracts it from the whole DL histogram
    d = d - d[:12].mean()  #ony pulse shape remains. rest is 0

    X = np.c_[np.ones(len(BG_IDX)), d[BG_IDX]] #X is just the table the fit needs

    #straight line fit using only the signal-free bins
    #asks: in those bins, which values of a and b make a + b·d match the sample best???
    a, b = np.linalg.lstsq(X, r[BG_IDX], rcond=None)[0] #sample count in bin  ≈  a × 1  +  b × (DL value in bin)
    #"which a and b make X times [a, b] come closest to the sample's counts in those bins?"

    return r - (a + b * d) #remove fitted bg


#the cleaned histograms from all the pixels are added bin by bin to make one combined histogram
def clean_file(path, dl_zones, n_pulses_dl):
    z, n_pulses = read_hlong(path)
    n_pulses = n_pulses or n_pulses_dl
    if dl_zones is None:                                      # no DL: "exp" or "flat" background
        return sum(clean_zone(z[k], None, n_pulses) for k in z if use_zone(k)), n_pulses
    if n_pulses != n_pulses_dl:
        print(f"WARNING {os.path.basename(path)}: {n_pulses} pulses vs DL {n_pulses_dl} - use a DL with the same iterations")
    y = sum(clean_zone(z[k], dl_zones[k], n_pulses) for k in z if k in dl_zones and use_zone(k))
    return y, n_pulses


def print_dl_counts(dl_zones, n_pulses, path, label="DL"):
    """Print what the DL file contains: average counts per frame, summed over all zones. Shows the pre-pulse floor,
    the crosstalk peak and the pedestal - the shape the background model scales and subtracts."""
    d = sum(dl_zones.values())
    floor = d[:12].mean()
    k = np.arange(len(d))
    pk = 12 + int(np.argmax(d[12:30]))
    print(f"{label} counts ({os.path.basename(path)}, sum of {len(dl_zones)} zones, per frame, {n_pulses} pulses):")
    print(f"   floor (bins 0-11)      {floor:9.1f} per bin")
    print(f"   crosstalk peak (bin {pk:d}) {d[pk]:9.1f}  = floor + {d[pk]-floor:.1f}")
    for b in (13, 20, 30, 40, 60, 100):
        if b < len(d):
            print(f"   bin {b:3d}                {d[b]:9.1f}  = floor + {d[b]-floor:.1f}")
    print(f"   total above floor, bins 12-40: {np.sum(d[12:41]-floor):.0f}   whole histogram: {d.sum():.0f}")


# -----------------------------------------------------------------------------
WIN_PRE, WIN_POST = 3, 25    # window = IRF peak - WIN_PRE ... IRF peak + WIN_POST bins; change with --win-pre/--win-post


WIN_SHIFT = 0     # --hocal: bins between the HOCAL peak and the data window (0 for the light pipe; ~28 for the fiber)


def observables(y, irf, n_pulses, pre=None, post=None):
    pre = WIN_PRE if pre is None else pre
    post = WIN_POST if post is None else post
    k = int(np.argmax(irf))
    def mean(v, a, b):
        tt = (np.arange(a, b) + 0.5) * DT_PS #convert bins ps
        w = v[a:b]; return (w * tt).sum() / w.sum() #weighted avg of phtons in window: i.e. centre of mass , barycentre
    ra, rb = max(0, k - pre), min(len(irf), k + post)            # window on the time reference (IRF or HOCAL)
    a, b = max(0, k + WIN_SHIFT - pre), min(len(y), k + WIN_SHIFT + post)   # window on the data
    #time delay between data and reference , intensity ( counts / pulses ). A fixed WIN_SHIFT only adds a constant
    #to the delay, which t0 absorbs.
    return mean(y, a, b) - mean(irf, ra, rb), y[a:b].sum() / (n_pulses / 1000.0)


# -----------------------------------------------------------------------------
#diffusion model (solution of the diffusion eqn) from Contini 97
def _A(n):
    t1 = 4*(-1-n**2+6*n**3-10*n**4-3*n**5+2*n**6+6*n**7-3*n**8-(6*n**2+9*n**6)*(n**2-1)**0.5)/(3*n*(n**2-1)**2*(n**2+1)**3)
    t2 = (-8+28*n**2+35*n**3-140*n**4+98*n**5-13*n**7+13*n*(n**2-1)**3*(1-(1/n**2))**0.5)/(105*n**3*(n**2-1)**2)
    t3 = 2*n**3*(3+2*n**4)*np.log(((n-(1+n**2)**0.5)*(2+n**2+2*(1+n**2)**0.5)*(n**2+(n**4-1)**0.5))/(n**2*(n+(1+n**2)**0.5)*(-n**2+(n**4-1)**0.5)))/((n**2-1)**2*(n**2+1)**3.5)
    t4 = ((1+6*n**4+n**8)*np.log((-1+n)/(1+n))+4*(n**2+n**6)*np.log((n**2*(1+n))/(n-1)))/((n**2-1)**2*(n**2+1)**3)
    B = 1+1.5*(2*(1-1/n**2)**1.5/3+t1+t2+((1+6*n**4+n**8)*(1-(n**2-1)**1.5/n**3))/(3*(n**4-1)**2)+t3)
    C = 1-((2+2*n-3*n**2+7*n**3-15*n**4-19*n**5-7*n**6+3*n**7+3*n**8+3*n**9)/(3*n**2*(n-1)*(n+1)**2*(n**2+1)**2))-t4
    return B / C

#constants
N_TISSUE = 1.41
A_EBC = _A(N_TISSUE)
TT = np.arange(1.0, 20000.0, 2.0)       # ps


def tpsf(rho_mm, mua_cm, musp_cm, n=None):
    """Contini '97 eq. 36, semi-infinite, absolute units up to a constant. TT in ps."""
    mua, musp = mua_cm / 10, musp_cm / 10
    n = N_TISSUE if n is None else n
    A = A_EBC if n == N_TISSUE else _A(n)             # boundary term depends on the refractive index mismatch
    v = C_MM_PS / n; D = 1 / (3 * musp); z0 = 1 / musp; ze = 2 * A * D
    z3, z4 = -z0, 2 * ze + z0
    t = TT
    return -np.exp(-mua*v*t - rho_mm**2/(4*D*v*t)) / (2*(4*np.pi*D*v)**1.5 * t**2.5) * \
           (z3*np.exp(-z3**2/(4*D*v*t)) - z4*np.exp(-z4**2/(4*D*v*t)))

#observables is now mean arrival time and total counts so find this for model curves
def obs_n(obs):
    """Refractive index carried by an observation (dT, I[, n]); default N_TISSUE."""
    return obs[2] if len(obs) > 2 else N_TISSUE


def model_obs(mua, musp, rho, n=None):
    r = tpsf(rho, mua, musp, n); s = r.sum()
    return (r * TT).sum() / s, s          # mean time (ps), CW intensity (arb.)


# ----------------------------------------------------------------------------- fitting
#this would need to be measured
# Error bars = PRECISION (repeatability), measured on B3: 3 repositioned repeats on 6 Oct + 17 Sep + 31 Aug.
SIG_T = 6.0      # ps  - SD of the mean arrival time between repositioned repeats, same session
SIG_I = 0.03     # 3 % - SD of the intensity between repeats (same within a day and between days)
# Accuracy vs MilanFIT (leave-one-out) would be --sig-t 40 --sig-i 0.5 (0.12 for index 2-3); not used by default.



def residual(mua, musp, rho, t0, logk, obs):
    dT, I = obs[0], obs[1]
    m, s = model_obs(mua, musp, rho, obs_n(obs))
    return [(dT - (m + t0)) / SIG_T, (np.log(I) - (logk + np.log(s))) / SIG_I]


def calibrate(obs, truth, refs, rho=18.0):
    """Calibrate t0 and logK with rho fixed (default: the measured 18 mm). rho=None also fits an effective rho."""

    #FIXED DISTANCE
    if rho is not None:
        F = lambda p: np.concatenate([residual(truth[n][0], truth[n][1], rho, p[0], p[1], obs[n]) for n in refs])

        #for each reference, log of measured intensity minus log of model intensity, averaged
        lk0 = np.mean([np.log(obs[n][1]) - np.log(model_obs(truth[n][0], truth[n][1], rho, obs_n(obs[n]))[1])
                       for n in refs])

        #measured counts  =  K × model intensity

        #least_squares adjusts t0 and logK, starting from −200 ps and lk0, until the mismatches are as small as possible
        r = least_squares(F, [-200.0, lk0])
        return np.array([rho, r.x[0], r.x[1]])

    def F(p): #p0 = rho , p1 = t0 , p2 = logK
        return np.concatenate([residual(truth[n][0], truth[n][1], p[0], p[1], p[2], obs[n]) for n in refs]) #compare ref with obs
    best = None

    #FIT DISTANCE : NOT RELEVANT HERE
    for r0 in (12.0, 16.0, 20.0):
        lk0 = np.mean([np.log(obs[n][1]) - np.log(model_obs(truth[n][0], truth[n][1], r0, obs_n(obs[n]))[1])
                       for n in refs])
        r = least_squares(F, [r0, -150.0, lk0], bounds=([5, -1500, -np.inf], [40, 1500, np.inf]))


        if best is None or r.cost < best.cost:
            best = r
    return best.x #return best result


MUSP_FIX_FROM = "both"   # with a fixed musp: fit mua from "t" (mean time), "I" (intensity) or "both"


def _fit_mua_1obs(o, rho, t0, logk, musp, k, x0=None):
    """mua from ONE observable (k = 0 mean time, 1 intensity) with musp fixed. One equation, one unknown: the
    model is matched exactly, so the answer does not depend on --sig-t / --sig-i (only its SD does)."""
    F = lambda p: residual(p[0], musp, rho, t0, logk, o)[k:k + 1]
    best = None
    for s in ((x0,) if x0 is not None else (0.02, 0.1, 0.3, 1.0)):
        r = least_squares(F, [s], bounds=([0.002], [3]))
        if best is None or r.cost < best.cost:
            best = r
    if x0 is not None and (best.cost > 1e-6 or best.x[0] <= 0.0021 or best.x[0] >= 2.99):
        return _fit_mua_1obs(o, rho, t0, logk, musp, k)          # warm start got stuck -> full multi-start
    J = best.jac
    try:
        sd = float(np.sqrt(np.linalg.inv(J.T @ J))[0, 0])
    except np.linalg.LinAlgError:
        sd = np.nan
    return best.x[0], sd


def invert_fixed(o, rho, t0, logk, musp, x0=None):
    """Fit mua alone with musp held at `musp` (cm-1), from the observable(s) chosen by MUSP_FIX_FROM.
    'both' = plain average of the timing-only and the intensity-only mua (equal weight, NOT weighted by
    --sig-t / --sig-i), so the recovered value never depends on the sigmas. Its SD = SD of that average.
    Returns ([mua, musp], [mua_sd, 0]) like invert(); the musp SD is 0 because it is not fitted."""
    if MUSP_FIX_FROM in ("t", "I"):
        mua, sd = _fit_mua_1obs(o, rho, t0, logk, musp, 0 if MUSP_FIX_FROM == "t" else 1, x0)
    else:
        mt, st = _fit_mua_1obs(o, rho, t0, logk, musp, 0, x0)
        mi, si = _fit_mua_1obs(o, rho, t0, logk, musp, 1, x0)
        mua, sd = 0.5 * (mt + mi), 0.5 * np.hypot(st, si)
    return np.array([mua, musp]), [sd, 0.0]


def invert(o, rho, t0, logk, musp_fix=None):
    if musp_fix is not None:
        return invert_fixed(o, rho, t0, logk, musp_fix)
    best = None

    #initial guesses
    for s in ((0.1, 5), (0.1, 10), (0.3, 8), (0.05, 15), (0.4, 5), (0.3, 15)):

        #The solver adjusts μa and μs′ until the model's mean time and intensity match the measured ones!
        r = least_squares(lambda p: residual(p[0], p[1], rho, t0, logk, o), s, bounds=([0.002, 0.5], [3, 50]))


        if best is None or r.cost < best.cost:
            best = r

    #for error bars
    J = best.jac
    try:
        cov = np.linalg.inv(J.T @ J); err = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        err = [np.nan, np.nan]


    return best.x, err



# ----------------------------------------------------------------------------- time series (frame by frame)
# In 4x4 mode the sensor measures the 16 zones in two halves (zones 1-8, then 9-16). Every logged record
# contains all 16 #HLONG rows, but only one half is new; the other half is repeated from the previous record.
# So one full, independent frame = one new update of every zone. read_hlong_frames keeps only the new
# updates of each zone and pairs them up into full frames.

def read_hlong_frames(path):
    """Return (frames, t_s, iterations).
    frames: {zone: array (n_frames, n_bins)} with only genuinely new histograms,
    t_s: time of each frame in seconds from the start of the file (mean of its zones' timestamps)."""
    upd, last, iters, ts = {}, {}, None, None
    with open(path, errors="ignore") as fh:
        for line in fh:
            if line.startswith(("#ITT", "#OBJ", "#TMP", "#LIGHT", "#RES")):
                p = split_fields(line)
                if len(p) > 1:
                    try:
                        ts = float(p[1]) / 1000.0                # record timestamp, ms -> s
                    except ValueError:
                        pass                                     # header row
                if line.startswith("#ITT") and iters is None and len(p) > 2 and p[2].strip().isdigit():
                    iters = int(p[2]) * 1000
            elif line.startswith(("#HLONG", "#HOCAL0", "#HOCAL1")):
                p = split_fields(line)
                if len(p) > 10 and p[1].strip().lstrip("-").isdigit():
                    h = np.array([float(x) for x in p[1:] if x.strip() != ""])
                    z = p[0]
                    if z not in last or not np.array_equal(h, last[z]):   # only keep new updates
                        upd.setdefault(z, []).append((ts if ts is not None else np.nan, h))
                    last[z] = h
    if not upd:
        return {}, np.array([]), iters
    n = min(len(v) for v in upd.values())
    if n == 0:
        return {}, np.array([]), iters
    frames = {z: np.array([h for _, h in v[:n]]) for z, v in upd.items()}
    TS = np.array([[tt for tt, _ in v[:n]] for v in upd.values()], dtype=float)
    ok = np.isfinite(TS).any(axis=0)
    t = np.full(n, np.nan)
    t[ok] = np.nanmean(TS[:, ok], axis=0)
    idx = np.arange(n)
    good = np.isfinite(t)
    if good.sum() >= 2 and np.ptp(t[good]) > 0 and np.all(np.diff(t[good]) >= 0):
        # usable timestamps; fill the odd frame without one (e.g. frames logged before the first timestamp line)
        if not good.all():
            period = np.median(np.diff(t[good]) / np.diff(idx[good]))
            t = np.interp(idx, idx[good], t[good], left=np.nan, right=np.nan)
            first, last = idx[good][0], idx[good][-1]
            t[:first] = t[first] - (first - idx[:first]) * period
            t[last + 1:] = t[last] + (idx[last + 1:] - last) * period
        t = t - t[0]
    else:
        # Excel re-saves turn timestamps into e.g. 1.75E+12 (the ms are lost) - then use the frame number instead
        print(f"WARNING {os.path.basename(path)}: timestamps unusable (file re-saved by Excel, or missing) - "
              f"using frame number x {FRAME_PERIOD_S} s")
        t = idx * FRAME_PERIOD_S
    return frames, t, iters


FRAME_PERIOD_S = 0.46        # used only when a file has no usable timestamps


def clean_frames(path, dl_zones, n_pulses_dl, group=1):
    """Clean each frame like clean_file does for the average. group>1 sums that many consecutive frames
    (more counts per point, fewer points)."""
    frames, t, n_pulses = read_hlong_frames(path)
    n_pulses = n_pulses or n_pulses_dl
    nf = len(t) // group
    taus = {}
    if BG_MODEL == "exp":     # pedestal decay constants fitted once per zone on the file average, reused per frame
        taus = {k: _fit_ped_taus(as_counts(frames[k].mean(axis=0))) for k in frames}
    Ys, T, Hs = [], [], []
    hk = [k for k in ("#HOCAL0", "#HOCAL1") if k in frames]
    for i in range(nf):
        sl = slice(i * group, (i + 1) * group)
        if hk:                                   # reference histogram of the same frames (for --hocal)
            Hs.append(clean_hocal(np.mean([frames[k][sl].mean(axis=0) for k in hk], axis=0)))
        # average the raw frames in the group first, then clean
        y = sum(clean_zone(frames[k][sl].mean(axis=0), None if dl_zones is None else dl_zones[k], n_pulses,
                           taus.get(k)) for k in frames if (dl_zones is None or k in dl_zones) and use_zone(k))
        Ys.append(y * group)                     # back to total counts for the group
        T.append(t[sl].mean())
    clean_frames.hocal = np.array(Hs) if Hs else None
    return np.array(Ys), np.array(T), n_pulses * group


MIN_SIGNAL = 0.2    # time points with total counts below this fraction of the file's median are left blank (--min-signal)


def fit_timeseries(path, irf, dl_zones, n_pulses_dl, cal, group=1, quiet=False, n_idx=None, musp_fix=None):
    """Fit every frame (or group of frames) of one file with a fixed calibration (rho, t0, logK).
    Returns a structured list of rows: frame, t_s, dT_ps, I, mua, mua_sd, musp, musp_sd."""
    rho, t0, lk = cal
    Ys, T, npul = clean_frames(path, dl_zones, n_pulses_dl, group)
    rows, prev, skipped = [], None, 0
    if TREF == "hocal":
        H = clean_frames.hocal
        if H is None:
            raise SystemExit(f"--hocal: no #HOCAL rows in {os.path.basename(path)}")
        OBS = [observables(y, h, npul) for y, h in zip(Ys, H)]   # each frame against its own reference pulse
    else:
        OBS = [observables(y, irf, npul) for y in Ys]
    if n_idx is not None:
        OBS = [tuple(o) + (n_idx,) for o in OBS]
    I_all = np.array([o[1] for o in OBS], dtype=float)
    I_ref = np.nanmedian(I_all[I_all > 0]) if np.any(I_all > 0) else np.nan
    for i, (y, t) in enumerate(zip(Ys, T)):
        o = OBS[i]
        # a time point with no usable signal (probe lifted, dropout, start/end of a recording) has total counts that
        # are <= 0 or far below the rest of the file; it is kept as a gap (NaN) instead of being fitted to nonsense
        if not (np.isfinite(o[0]) and np.isfinite(o[1]) and o[1] > 0 and o[1] >= MIN_SIGNAL * I_ref):
            rows.append((i, t, o[0], o[1], np.nan, np.nan, np.nan, np.nan)); skipped += 1
            continue
        if musp_fix is not None:                 # fixed scattering: 1-parameter fit of mua
            x, err = invert_fixed(o, rho, t0, lk, musp_fix, None if prev is None else prev[0])
            prev = x
            rows.append((i, t, o[0], o[1], x[0], err[0], x[1], err[1]))
            continue
        if prev is not None:                     # start from the previous frame's answer (faster, smoother)
            r = least_squares(lambda p: residual(p[0], p[1], rho, t0, lk, o), prev, bounds=([0.002, 0.5], [3, 50]))
            x, J = r.x, r.jac
            if r.cost > 1.0:                     # fall back to the full multi-start search if it got stuck
                x, _ = invert(o, rho, t0, lk); J = None
        else:
            x, _ = invert(o, rho, t0, lk); J = None
        if J is None:
            J = least_squares(lambda p: residual(p[0], p[1], rho, t0, lk, o), x, bounds=([0.002, 0.5], [3, 50])).jac
        try:
            err = np.sqrt(np.diag(np.linalg.inv(J.T @ J)))
        except np.linalg.LinAlgError:
            err = [np.nan, np.nan]
        prev = x
        rows.append((i, t, o[0], o[1], x[0], err[0], x[1], err[1]))
    if skipped and not quiet:
        print(f"WARNING {os.path.basename(path)}: {skipped} of {len(rows)} time points had no usable signal "
              f"(counts below {MIN_SIGNAL:.0%} of the file's median - probe off/lifted?) and were left blank")
    return rows


def noise_sd(y):
    """Point-to-point noise of a time series: SD of the differences between neighbouring points / sqrt(2).
    Slow real changes (physiology, drift) hardly affect it, so it measures precision, not variation."""
    y = np.asarray(y, dtype=float)
    d = np.diff(y)
    d = d[np.isfinite(d)]
    return float(np.std(d) / np.sqrt(2)) if len(d) > 2 else np.nan


def local_noise_sd(y, window=25):
    """Point-to-point noise as it changes through a recording: noise_sd computed in a sliding window of
    `window` points (the noise grows when the signal drops, e.g. during an occlusion)."""
    y = np.asarray(y, dtype=float)
    d = np.r_[np.nan, np.diff(y)] ** 2 / 2.0          # squared half-differences, aligned with y
    return np.sqrt(moving_average(d, max(5, int(window))))


def block_precision(path, irf, dl_zones, n_pulses_dl, cal, npoints=50, n_idx=None, musp_fix=None):
    """Statistical (photon-noise) uncertainty of a file-averaged result. The recording is cut into about `npoints`
    consecutive pieces, each piece is fitted, and the point-to-point noise of that series (noise_sd, which ignores
    slow real changes such as physiology) is divided by sqrt(number of pieces): the standard error of the mean.
    Returns (nan, nan) if there are too few frames."""
    frames, t, _ = read_hlong_frames(path)
    g = max(1, len(t) // npoints)
    if len(t) // g < 5:
        return np.nan, np.nan
    R = np.array(fit_timeseries(path, irf, dl_zones, n_pulses_dl, cal, g, quiet=True, n_idx=n_idx,
                                    musp_fix=musp_fix), dtype=float)
    out = []
    for j in (4, 6):
        v = R[:, j]
        nv = np.sum(np.isfinite(v))
        out.append(noise_sd(v) / np.sqrt(nv) if nv >= 5 else np.nan)
    return tuple(out)


def moving_average(y, window):
    """Centred moving average over `window` points. window <= 1 returns y unchanged.
    Near the two ends the window shrinks to the points that exist, so the output has the same length as y."""
    y = np.asarray(y, dtype=float)
    if window <= 1 or len(y) < 2:
        return y.copy()
    k = np.ones(min(int(window), len(y)))
    ok = np.isfinite(y)                                   # blank (NaN) points are left out of the average
    num = np.convolve(np.where(ok, y, 0.0), k, mode="same")
    den = np.convolve(ok.astype(float), k, mode="same")
    out = np.full(len(y), np.nan)
    out[den > 0] = num[den > 0] / den[den > 0]
    return out


PHASES = ((None, 180, "rest"), (180, 360, "occlusion"), (360, None, "recovery"))   # protocol segments, seconds


def _phase_labels(ax, t, labels=True):
    """Dotted lines at 180 s and 360 s, with 'rest' / 'occlusion' / 'recovery' written just above each segment."""
    import matplotlib.transforms as mtransforms
    t = np.asarray(t, dtype=float); t = t[np.isfinite(t)]
    if t.size == 0:
        return
    t0, t1 = t.min(), t.max()
    tr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for a, b, lab in PHASES:
        a = t0 if a is None else a
        b = t1 if b is None else b
        lo, hi = max(a, t0), min(b, t1)
        if hi <= lo or not labels:
            continue                                            # recording does not reach this segment / no text
        ax.text(0.5 * (lo + hi), 1.01, lab, transform=tr, ha="center", va="bottom", fontsize=9, color="#3d3c38")
    for tm in (60, 180):
        if t0 < tm < t1:
            ax.axvline(tm, color="#6b6a64", ls=":", lw=1)


def plot_timeseries(rows, name, out_png, truth=None, smooth=1, show_noise=False):
    """mua (top) and musp (bottom) vs time for one sample. Grey dots = every time point; blue line = the moving
    average when smooth > 1; shaded band = +/- 1 noise SD (point-to-point noise, divided by sqrt(smooth) for the
    moving average)."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = np.array(rows, dtype=float); t = R[:, 1]
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True)
    for ax, j, lab, k in ((axs[0], 4, "μa (cm⁻¹)", 0), (axs[1], 6, "μs′ (cm⁻¹)", 1)):
        y = R[:, j]
        m, s = np.nanmean(y), np.nanstd(y)
        if not show_noise:
            # default (V2 style): the curve itself, no noise band
            title = f"mean {m:.3f}, frame-to-frame SD {s:.3f} ({100*s/m:.1f} %)"
            if smooth > 1:
                ys = moving_average(y, smooth)
                ax.plot(t, ys, color="#2a78d6", marker="o", lw=1.8, label=f"moving average ({smooth} points)")
                title += f", after smoothing SD {np.nanstd(ys):.3f} ({100*np.nanstd(ys)/m:.1f} %)"
            else:
                ax.plot(t, y, color="#2a78d6", lw=1.2, marker="o", ms=2.5, label="per frame")
        else:
            # --error: every raw point in light grey with the moving average on top (raw vs smoothed)
            title = f"mean {m:.3f}, frame-to-frame SD {s:.3f} ({100*s/m:.1f} %)"
            ax.plot(t, y, ".", color="#b9b8b1", ms=2.5, label="each point (raw)")
            if k == 0:
                ax.set_ylim(0.18, 0.26)
            if smooth > 1:
                ys = moving_average(y, smooth)
                ax.plot(t, ys, color="#2a78d6", lw=1.8, label=f"moving average ({smooth} points)")
                title += f", after smoothing SD {np.nanstd(ys):.3f} ({100*np.nanstd(ys)/m:.1f} %)"
        if truth is not None:
            ax.axhline(truth[k], color="#6b6a64", ls="--", lw=1, label="expected")
        _phase_labels(ax, t, labels=(k == 0))                   # rest / occlusion / recovery, above mua only
        ax.set_ylabel(lab); ax.grid(color="#ecebe6", lw=0.6)
       # ax.set_xlim(-10, 350)
        # (stats line above each panel removed - the numbers are printed in the console instead)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=7.5, loc="upper right", ncol=2)
    #axs[0].set_ylim(0.3, 0.45)                                  # mua axis range; musp is left automatic
    axs[1].set_xlabel("time (s)")
    fig.suptitle(name, x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


# ----------------------------------------------------------------------------- plots
SERIES_COL = ['#2a78d6', '#eb6834', '#1baf7a', '#c98500', '#e87ba4', '#008300', '#7d5ce0', '#6b6a64']
SERIES_MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color="#ecebe6", lw=0.6)


def plot_linearity(rows, truth, out_dir, tag="", stat=None, errbars="total"):
    """MEDPHOT-style linearity plots from the results table (needs certified values).
    mua: x = scattering series A-D, one line per absorption index.  musp: x = absorption index, one line per series.
    Solid = recovered (+/- sd), dashed = expected."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    stat = stat or {}
    def ebar(n, sys_sd, k):
        st = stat.get(n, (np.nan, np.nan))[k]
        st = 0.0 if not np.isfinite(st) else st
        return {"total": np.hypot(sys_sd, st), "stat": st, "sys": sys_sd, "none": 0.0}[errbars]
    R = {n.split("_")[0]: (mua, ebar(n, sm, 0), musp, ebar(n, ss, 1))
         for n, rho, t0, mua, sm, musp, ss in rows}   # "A2_2.0" plots as A2
    truth = {**truth, **{n.split("_")[0]: v for n, v in truth.items()}}
    ser = sorted({n[0] for n in R if len(n) == 2 and n in truth})
    idx = sorted({n[1] for n in R if len(n) == 2 and n in truth}, key=lambda c: int(c) if c.isdigit() else c)
    if not ser or not idx:
        print("plot_linearity: no samples with certified values named like A2 ... D7"); return
    os.makedirs(out_dir, exist_ok=True)
    for which, groups, xs, xlab, leg, ylab, k in (
            ("mua", idx, ser, "scattering series", "μa index {}", "μa (cm⁻¹)", 0),
            ("musp", ser, idx, "absorption index", "series {}", "μs′ (cm⁻¹)", 2)):
        fig, ax = plt.subplots(figsize=(6.6, 5.4)); xi = np.arange(len(xs))
        ymax = 0
        for gi, g in enumerate(groups):
            names = [(x + g) if which == "mua" else (g + x) for x in xs]
            ok = [i for i, n in enumerate(names) if n in R and n in truth]
            if not ok:
                continue
            x = xi[ok]; y = [R[names[i]][k] for i in ok]; e = [R[names[i]][k + 1] for i in ok]
            c = [truth[names[i]][0 if which == "mua" else 1] for i in ok]
            col, mk = SERIES_COL[gi % 8], SERIES_MK[gi % 8]
            ax.plot(x, c, color=col, lw=1.2, ls="--", alpha=0.8)
            ax.plot(x, y, color=col, lw=1.8)
            if errbars == "none":
                ax.plot(x, y, mk, ms=7, color=col, mec="white", mew=1.5)                       # markers only
            else:
                ax.errorbar(x, y, yerr=e, fmt=mk, ms=7, color=col, mec="white", mew=1.5, elinewidth=1.2, capsize=4, capthick=1.2)
            ax.plot([], [], mk, ms=7, color=col, mec="white", mew=1.5, label=leg.format(g))   # legend entry without caps
            ymax = max(ymax, max(y) + max(e), max(c))
        ax.plot([], [], color="#6b6a64", ls="--", lw=1.2, label="expected")
        ax.plot([], [], color="#6b6a64", lw=1.8, label="recovered")
        ax.set_xticks(xi); ax.set_xticklabels(xs); ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_ylim(0, ymax * 1.25)
        ax.set_title(f"{ylab.split(' ')[0]} vs expected {tag}".strip(), fontsize=11, loc="left")
        _style(ax); ax.legend(frameon=False, fontsize=8.5, loc="upper left", ncol=2)
        fn = os.path.join(out_dir, f"linearity_{which}{'_' + tag if tag else ''}.png")
        fig.tight_layout(); fig.savefig(fn, dpi=140); plt.close(fig)
        print("wrote", fn)


def plot_timeseries_overview(ts, out_png, truth=None, relative=False, smooth=1):
    """mua (top) and musp (bottom) of all time-series samples on one figure, as lines with point markers.
    relative=True plots % change from each sample's own mean, which makes changes comparable between very
    different phantoms. smooth > 1 plots the moving average."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 1, figsize=(9, 6.8), sharex=True)
    t_all = np.concatenate([np.array(r, dtype=float)[:, 1] for r in ts.values()])
    for ax, j, lab in ((axs[0], 4, "μa"), (axs[1], 6, "μs′")):
        for i, (n, rows) in enumerate(ts.items()):
            R = np.array(rows, dtype=float)
            col = SERIES_COL[i % 8] if len(ts) <= 8 else None
            y = R[:, j]
            if relative:
                y = 100 * (y / np.nanmean(y) - 1)
            ax.plot(R[:, 1], moving_average(y, smooth), lw=1.2, color=col, marker=SERIES_MK[i % 8], ms=3,
                    label=n)
        ax.set_ylabel(f"{lab} change (%)" if relative else f"{lab} (cm⁻¹)")
        _style(ax)
        _phase_labels(ax, t_all, labels=(j == 4))           # rest / occlusion / recovery text above mua only
    axs[1].set_xlabel("time (s)")
    axs[0].legend(frameon=False, fontsize=8, ncol=min(len(ts), 6), loc="lower left", bbox_to_anchor=(0, 1.07))
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)
    print("wrote", out_png)


# ----------------------------------------------------------------------------- --compare
def read_compare(path):
    """CSV of known phantoms -> {name: (mua, musp)} in cm-1. Separator ',' ';' or tab. Columns are found by their
    header (name/sample, a header containing 'mua', and one containing 'musp' / 'mus'); without a header the
    columns are taken as name, mua, musp."""
    out = {}
    lines = [l for l in open(path, encoding="utf-8-sig", errors="ignore") if l.strip() and not l.startswith("#")]
    if not lines:
        raise SystemExit(f"--compare: {path} is empty")
    head = [h.strip().lower() for h in _SPLIT.split(lines[0].strip())]
    def num(x):
        try:
            return float(x)
        except ValueError:
            return None
    has_header = num(head[1] if len(head) > 1 else "") is None
    i_n, i_a, i_s, i_r = 0, 1, 2, None
    if has_header:
        for i, h in enumerate(head):
            if i == 0:
                continue
            if "mua" in h or "absorption" in h:
                i_a = i
            elif "musp" in h or "mus" in h or "scatter" in h:
                i_s = i
            elif h.split(" ")[0] in ("n", "ri", "n_index", "nindex") or "refractive" in h:
                i_r = i
        lines = lines[1:]
    elif len(head) >= 4:
        i_r = 3                                   # no header: name, mua, musp, n
    for l in lines:
        p = [x.strip() for x in _SPLIT.split(l.strip())]
        if len(p) <= max(i_n, i_a, i_s):
            continue
        a, b = num(p[i_a]), num(p[i_s])
        if p[i_n] and a is not None and b is not None:
            out[p[i_n]] = (a, b)
            r = num(p[i_r]) if (i_r is not None and len(p) > i_r and p[i_r]) else None
            if r is not None:
                NIDX[p[i_n]] = r
    if not out:
        raise SystemExit(f"--compare: no rows with a name, mua and musp found in {path}")
    return out


NIDX = {}    # per-phantom refractive index from the compare CSV ('n' column); others use --n


def natural_key(name):
    """Sort 'idk2' before 'idk10' (numbers compared as numbers, not as text)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def match_name(n, known):
    """Sample file name -> name in the compare list: exact, else the part before the first '_' ('idk1_2' -> 'idk1')."""
    if n in known:
        return n
    base = n.split("_")[0]
    return base if base in known else None


def plot_compare(rows, comp, out_png, errbars="none", refs=()):
    """For every sample in the compare list: recovered value (dot) and expected value (cross), side by side,
    mua on the top panel and musp on the bottom. x axis = phantom name."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pts = []
    for n, rho, t0, mua, sm, musp, ss in rows:
        k = match_name(n, comp)
        if k is not None and n not in refs:                 # references agree by construction - left out
            pts.append((n, comp[k][0], mua, sm, comp[k][1], musp, ss))
    if not pts:
        print("--compare: none of the fitted samples (other than the references) is in the compare list")
        return []
    order = {k: i for i, k in enumerate(comp)}                 # the row order of the CSV
    pts.sort(key=lambda p: (order[match_name(p[0], comp)], natural_key(p[0])))
    RES, EXP = "#d6457e", "#2a78d6"                      # result = pink dot, expected = blue cross
    x = np.arange(len(pts))
    fig, axs = plt.subplots(2, 1, figsize=(max(6.0, 0.75 * len(pts) + 2.5), 7.2), sharex=True)
    for ax, (ie, ir, isd, lab) in zip(axs, ((1, 2, 3, "μa (cm⁻¹)"), (4, 5, 6, "μs′ (cm⁻¹)"))):
        e = np.array([p[ie] for p in pts]); r = np.array([p[ir] for p in pts]); sd = np.array([p[isd] for p in pts])
        ax.plot(x, e, "x", ms=10, mew=2.2, color=EXP, label="expected", zorder=3)
        if errbars != "none":
            ax.errorbar(x, r, yerr=sd, fmt="o", ms=8, color=RES, mec="white", mew=1.2, elinewidth=1.2,
                        capsize=4, capthick=1.2, zorder=4)
        else:
            ax.plot(x, r, "o", ms=8, color=RES, mec="white", mew=1.2, zorder=4)
        ax.plot([], [], "o", ms=8, color=RES, mec="white", mew=1.2, label="result")
        ax.set_ylabel(lab); ax.set_ylim(0, 1.25 * max(e.max(), (r + (sd if errbars != "none" else 0)).max()))
        _style(ax)
        ax.legend(frameon=False, fontsize=9, loc="best")
    names = [p[0] for p in pts]
    axs[1].set_xticks(x); axs[1].set_xticklabels(names, rotation=0 if len(pts) <= 10 else 45,
                                                 ha="center" if len(pts) <= 10 else "right")
    axs[1].set_xlim(-0.6, len(pts) - 0.4)
    axs[1].set_xlabel("phantom")
    fig.suptitle("recovered vs expected", x=0.01, ha="left", fontsize=11)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.tight_layout(); fig.savefig(out_png, dpi=140); plt.close(fig)
    print("wrote", out_png)
    return pts


def plot_compare_tiles(pts, out_png, subtitle="", per_row=5):
    """One tile per phantom with 'mua / musp' error vs expected (%). Colour = the worse of the two:
    green <= 20 %, amber <= 50 %, red > 50 %. Side panel counts phantoms per band."""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    BANDS = [(20, "within 20 %", "#dff0ec", "#2a9d8f"), (50, "within 50 %", "#fcebd7", "#b26b00"),
             (np.inf, "worse than 50 %", "#f6dcdc", "#d24b4b")]
    def band(worst):
        return next(i for i, b in enumerate(BANDS) if worst <= b[0])
    n = len(pts); ncol = min(per_row, n); nrow = int(np.ceil(n / ncol))
    tw, th, gap = 1.7, 1.0, 0.15                       # tile size (inches)
    side = 2.6
    W = max(ncol * (tw + gap) + side + 0.6, 10.0); H = max(nrow * (th + gap), 2.95) + 1.3
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    ax.text(0.3, H - 0.35, "μa / μs′ error vs expected (%)", fontsize=12, color="#24318c", weight="bold", va="center")
    counts = [0, 0, 0]
    for i, (name, ea, ra, _, es, rs, _) in enumerate(pts):
        da, ds = 100 * (ra / ea - 1), 100 * (rs / es - 1)
        b = band(max(abs(da), abs(ds))); counts[b] += 1
        _, _, bg, fg = BANDS[b]
        r, c = divmod(i, ncol)
        x0 = 0.3 + c * (tw + gap); y0 = H - 0.75 - (r + 1) * (th + gap) + gap
        ax.add_patch(FancyBboxPatch((x0, y0), tw, th, boxstyle="round,pad=0,rounding_size=0.06",
                                    fc=bg, ec="none"))
        ax.text(x0 + tw / 2, y0 + th * 0.70, name, ha="center", va="center", fontsize=10, color=fg, weight="bold")
        ax.text(x0 + tw / 2, y0 + th * 0.36, f"{da:+.0f} / {ds:+.0f}".replace("-", "−"), ha="center", va="center",
                fontsize=11, color="#2b2a27")
        at = []
        if ra <= 0.0021 or ra >= 2.99: at.append("μa at bound")
        if rs <= 0.51 or rs >= 49.9: at.append("μs′ at bound")
        if at:
            ax.text(x0 + tw / 2, y0 + th * 0.13, ", ".join(at), ha="center", va="center", fontsize=7.5,
                    color=fg, style="italic")
    # side panel
    sx = 0.3 + ncol * (tw + gap) + 0.25; sw = side - 0.1
    top = H - 0.75; ph = 0.55 + 3 * 0.75 + 0.1
    ax.add_patch(FancyBboxPatch((sx, top - ph), sw, ph, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="#f3f6fb", ec="none"))
    ax.text(sx + 0.15, top - 0.25, "Summary", fontsize=11, color="#24318c", weight="bold", va="center")
    for j, (lim, lab, bg, fg) in enumerate(BANDS):
        y = top - 0.55 - (j + 1) * 0.75 + 0.1
        ax.add_patch(FancyBboxPatch((sx + 0.1, y), sw - 0.2, 0.62, boxstyle="round,pad=0,rounding_size=0.05",
                                    fc=bg, ec="none"))
        ax.text(sx + 0.35, y + 0.31, str(counts[j]), fontsize=20, color=fg, weight="bold", va="center")
        ax.text(sx + 0.9, y + 0.31, lab, fontsize=9.5, color="#2b2a27", weight="bold", va="center")
    ax.text(0.3, 0.25, (subtitle + "  ·  " if subtitle else "") + "Green ≤ 20 %, amber ≤ 50 %, red > 50 % "
            "(worse of μa, μs′). References not shown.", fontsize=8, color="#6b6a64", style="italic", va="center")
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=140); plt.close(fig)
    print("wrote", out_png)


# ----------------------------------------------------------------------------- CLI
def main():

    #arguments for cmd prompt
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--dl", default=None,
                    help="direct-light (DL) file for the background. Can also be a folder of DL files (or a "
                         "comma-separated list): then each sample and reference uses the DL recorded closest in time "
                         "to it - for phantoms from different sessions in one folder. If the folder is --data itself, "
                         "files whose name starts with 'DL' are the DLs")
    ap.add_argument("--bg-model", choices=["dl", "exp", "flat"], default=None,
                    help="background: dl = offset + scaled DL pedestal (default when --dl is given); "
                         "exp = offset + exponential pedestal fitted on the background bins (no DL needed); "
                         "flat = constant")
    ap.add_argument("--bg-bins", default="0-11,70-127",
                    help="signal-free bins used to fit the background, e.g. '0-12,18-37,76-127' (inclusive ranges)")
    ap.add_argument("--ped-start", type=int, default=13, help="first bin of the crosstalk pedestal (for --bg-model exp)")
    ap.add_argument("--win-pre", type=int, default=3, help="window starts this many bins before the IRF peak")
    ap.add_argument("--win-post", type=int, default=25, help="window ends this many bins after the IRF peak")
    ap.add_argument("--t-range", default=None,
                    help="time series: keep only this time window in seconds, e.g. '0-428' (cuts off the probe being "
                         "removed at the end, or setup at the start)")
    ap.add_argument("--min-signal", type=float, default=0.2,
                    help="time series: leave blank any point whose total counts are below this fraction of the "
                         "file's median (probe lifted / off the skin). 0 keeps everything with counts > 0")
    ap.add_argument("--zones", default=None,
                    help="use only these pixels/zones, e.g. '6,7,10,11' (default: all 16)")
    ap.add_argument("--frame-period", type=float, default=0.46,
                    help="seconds per frame, used only for files whose timestamps are unusable")
    ap.add_argument("--irf", default="irf_10mm", help="IRF file name in --data (without .csv). Not used with --hocal")
    ap.add_argument("--hocal", action="store_true",
                    help="use each file's own reference histogram (#HOCAL) as time zero instead of an IRF file. "
                         "No IRF measurement needed; files whose name starts with 'irf' are then ignored")
    ap.add_argument("--compare", default=None,
                    help="CSV of phantoms with known properties: columns name, mua, musp (cm-1). Any sample whose name "
                         "matches (exactly, or the part before the first '_') is plotted recovered vs expected")
    ap.add_argument("--win-shift", type=int, default=None,
                    help="--hocal only: bins from the HOCAL peak to the start of the data signal. Found automatically "
                         "(0 for the light pipe, ~28 for the fiber, whose signal arrives later than the HOCAL pulse)")
    ap.add_argument("--n", type=float, default=1.41,
                    help="refractive index of the samples (default 1.41). A phantom with a different index can be "
                         "given its own value in an 'n' column of the --compare CSV")
    ap.add_argument("--refs", default="B2,B3,C2,C3,D2")
    ap.add_argument("--truth", default=None); ap.add_argument("--out", default="moments_fit_results.csv")
    ap.add_argument("--loo", action="store_true")
    ap.add_argument("--rho", type=float, default=18.0, help="source-detector separation in mm (fixed)")
    ap.add_argument("--fit-rho", action="store_true", help="fit an effective rho instead of fixing it")
    ap.add_argument("--self-ref", default=None, help="reference(s) used to evaluate the --refs phantoms themselves")
    ap.add_argument("--ref-dir", default=None,
                    help="folder that holds the reference phantom files (--refs). When set, references are read ONLY "
                         "from here, and every csv in --data is treated as a sample")
    ap.add_argument("--ref-irf", default=None,
                    help="IRF file name (without .csv) inside --ref-dir for the reference session. "
                         "Default: use the IRF from --data")
    ap.add_argument("--ref-dl", default=None, help="DL file for the reference session. Default: the --dl file")
    ap.add_argument("--outdir", "--out-dir", dest="outdir", default=None,
                    help="folder for everything the script saves. --out, --ts-dir and --plot-dir are placed inside it "
                         "(unless given as absolute paths). Created if it does not exist. Default: current folder")
    ap.add_argument("--timeseries", default=None,
                    help="fit frame by frame: comma-separated sample names, or 'all'")
    ap.add_argument("--group", type=int, default=1, help="sum this many consecutive frames per time point")
    ap.add_argument("--smooth", type=int, default=1,
                    help="moving-average window, in time points, applied to the fitted mua time series (1 = off)")
    ap.add_argument("--ts-dir", default="timeseries", help="folder for the time-series CSVs and plots")
    ap.add_argument("--no-plot", action="store_true", help="do not save per-sample time-series plots")
    ap.add_argument("--ts-errors", "--error", dest="ts_errors", action="store_true",
                    help="time-series plots with every raw point in light grey and the moving average on top "
                         "(default plot: just the curve / moving average)")
    ap.add_argument("--plot-linearity", action="store_true",
                    help="save MEDPHOT-style linearity plots (mua and musp vs expected) from the results")
    ap.add_argument("--plot-dir", default="plots", help="folder for the linearity / overview plots")
    ap.add_argument("--relative", action="store_true", help="time-series overview as %% change from each sample's mean")
    ap.add_argument("--err-blocks", type=int, default=0,
                    help="optional photon-noise error of each averaged result (off by default; e.g. 50 cuts the "
                         "recording into ~50 pieces and uses their point-to-point scatter)")
    ap.add_argument("--errbars", choices=["sys", "total", "stat", "none"], default="none",
                    help="error bars on the linearity plots: none (default) = markers only; sys = precision from --sig-t/--sig-i; total = calibration "
                         "and photon noise; stat = photon noise only (the last two need --err-blocks)")
    ap.add_argument("--sig-t", type=float, default=6.0,
                    help="uncertainty of the mean arrival time, ps. Default 6 = precision (B3 repositioned repeats, "
                         "same session). Use 40 for accuracy vs MilanFIT, or if the references are from another day")
    ap.add_argument("--sig-i", type=float, default=0.03,
                    help="uncertainty of the intensity, fraction. Default 0.03 = precision (B3 repeats, within and "
                         "between days). Use 0.5 (0.12 for index 2-3) for accuracy vs MilanFIT")
    ap.add_argument("--fix-musp", default=None,
                    help="fix musp and fit only mua: a number in cm-1 for every sample, or 'truth' for each sample's "
                         "certified / --compare musp. The reference calibration is unchanged")
    ap.add_argument("--fix-musp-from", choices=["t", "I", "both"], default="both",
                    help="with --fix-musp: fit mua from the mean time only (t), the intensity only (I), or both "
                         "(default: average of the two, independent of --sig-t / --sig-i)")
    a = ap.parse_args()
    global MUSP_FIX_FROM
    MUSP_FIX_FROM = a.fix_musp_from
    musp_const = None
    if a.fix_musp is not None and a.fix_musp.lower() != "truth":
        try:
            musp_const = float(a.fix_musp)
        except ValueError:
            ap.error("--fix-musp must be a number (cm-1) or 'truth'")
        if not 0.5 <= musp_const <= 50:
            ap.error("--fix-musp must be between 0.5 and 50 cm-1")
    if a.smooth < 1:
        ap.error("--smooth must be 1 or more")
    global BG_IDX, BG_MODEL, PED_START, WIN_PRE, WIN_POST, FRAME_PERIOD_S, MIN_SIGNAL, SIG_T, SIG_I
    global ZONES
    ZONES = parse_zones(a.zones) if a.zones else None
    if ZONES:
        print(f"zones: {', '.join(sorted(z[-2:] for z in ZONES))}")
    MIN_SIGNAL = a.min_signal; SIG_T = a.sig_t; SIG_I = a.sig_i
    global TREF, N_TISSUE, A_EBC
    TREF = "hocal" if a.hocal else "irf"
    N_TISSUE = a.n; A_EBC = _A(N_TISSUE)
    BG_IDX = parse_bins(a.bg_bins); PED_START = a.ped_start
    BG_MODEL = a.bg_model or ("dl" if a.dl else "exp")
    if BG_MODEL == "dl" and not a.dl:
        ap.error("--bg-model dl needs a --dl file (or use --bg-model exp)")
    WIN_PRE, WIN_POST, FRAME_PERIOD_S = a.win_pre, a.win_post, a.frame_period
    print(f"background: {BG_MODEL} on bins {a.bg_bins} | window: {'HOCAL' if a.hocal else 'IRF'} peak "
          f"-{WIN_PRE} ... +{WIN_POST} bins | time zero: {'each file own #HOCAL' if a.hocal else a.irf + '.csv'}")
    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        a.out = os.path.join(a.outdir, a.out)          # os.path.join keeps absolute paths as they are
        a.ts_dir = os.path.join(a.outdir, a.ts_dir)
        a.plot_dir = os.path.join(a.outdir, a.plot_dir)
        print(f"saving results to {os.path.abspath(a.outdir)}")
    out_parent = os.path.dirname(a.out)
    if out_parent:
        os.makedirs(out_parent, exist_ok=True)

    #load and clean data
    def as_file(path, what):
        """Accept a file, or a folder holding exactly one .csv; otherwise stop with a clear message."""
        if path is None or os.path.isfile(path):
            return path
        if os.path.isdir(path):
            csvs = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
            if len(csvs) == 1:
                print(f"{what}: using {csvs[0]} in folder {path}")
                return os.path.join(path, csvs[0])
            ap.error(f"{what} '{path}' is a folder with {len(csvs)} csv files - give the file itself")
        ap.error(f"{what} file not found: {path}")
    # --dl: one file (as before), or several - a folder of DL files, or a comma-separated list. With several,
    # every sample and reference uses the DL recorded closest in time to it (same iteration count preferred).
    def dl_paths(spec):
        if spec is None:
            return []
        out = []
        for part in [x for x in spec.split(",") if x.strip()] if not os.path.exists(spec) else [spec]:
            part = part.strip()
            if os.path.isdir(part):
                csvs = sorted(f for f in os.listdir(part) if f.lower().endswith(".csv"))
                if os.path.abspath(part) == os.path.abspath(a.data):       # DLs kept among the samples
                    csvs = [f for f in csvs if f.lower().startswith("dl")]
                out += [os.path.join(part, f) for f in csvs]
            elif os.path.isfile(part):
                out.append(part)
            else:
                ap.error(f"DL file or folder not found: {part}")
        return out
    dl_list = dl_paths(a.dl); rdl_list = dl_paths(a.ref_dl)
    multi_dl = len(dl_list) > 1 or len(rdl_list) > 1
    if (a.dl or a.ref_dl) and not dl_list + rdl_list:
        ap.error("no DL .csv files found in --dl / --ref-dl")
    a.dl = dl_list[0] if dl_list else None
    a.ref_dl = rdl_list[0] if rdl_list else None
    DL_LIB = []
    if multi_dl and BG_MODEL == "dl":
        for f in dict.fromkeys(dl_list + rdl_list):
            z, it = read_hlong(f)
            DL_LIB.append((f, start_time(f), it, z))
        print(f"{len(DL_LIB)} DL files - each file uses the DL recorded closest in time:")
        for f, t, it, _ in DL_LIB:
            print(f"   {os.path.basename(f):28s} {time.strftime('%Y-%m-%d %H:%M', time.localtime(t)) if t else 'no time':16s} "
                  f"{it} pulses")
    DL_USED = {}

    def dl_for(path, default):
        """DL (zones, pulses) for one data file: the given default, or with several DLs the nearest in time."""
        if not DL_LIB:
            return default
        t = start_time(path)
        it = read_hlong_iters(path)
        cand = [d for d in DL_LIB if d[2] == it] or DL_LIB
        if t is None or any(d[1] is None for d in cand):
            ap.error(f"several DL files given, but {os.path.basename(path)} or a DL has no usable timestamp "
                     f"(re-saved in Excel?) - give one DL with --dl instead")
        f, td, itd, z = min(cand, key=lambda d: abs(d[1] - t))
        DL_USED[os.path.basename(path)] = (os.path.basename(f), (t - td) / 60.0)
        return z, itd
    if not a.hocal and not os.path.isfile(os.path.join(a.data, a.irf + ".csv")):
        ap.error(f"IRF '{a.irf}.csv' not found in --data {a.data} (or use --hocal, which needs no IRF)")
    dl_z, dl_n = read_hlong(a.dl) if (a.dl and BG_MODEL == "dl") else (None, None)
    if dl_z is not None and not DL_LIB:
        print_dl_counts(dl_z, dl_n, a.dl)
    refs = [r for r in a.refs.split(",") if r]
    skip = {a.irf, os.path.splitext(os.path.basename(a.out))[0]}
    for f in dl_list + rdl_list:                                          # a DL file kept in --data is not a sample
        if os.path.dirname(os.path.abspath(f)) == os.path.abspath(a.data):
            skip.add(os.path.splitext(os.path.basename(f))[0])
    names = sorted((f[:-4] for f in os.listdir(a.data) if f.endswith(".csv") and f[:-4] not in skip
                    and not (a.hocal and f.lower().startswith("irf"))), key=natural_key)
    irf = None if a.hocal else clean_file(os.path.join(a.data, a.irf + ".csv"),
                                          *dl_for(os.path.join(a.data, a.irf + ".csv"), (dl_z, dl_n)))[0]
    comp = read_compare(a.compare) if a.compare else {}
    def n_of(name):
        k = match_name(name, NIDX)
        return NIDX[k] if k is not None else N_TISSUE
    global WIN_SHIFT
    WIN_SHIFT = 0
    if a.hocal:
        if a.win_shift is not None:
            WIN_SHIFT = a.win_shift
        else:
            # where is the signal relative to the HOCAL pulse? median over the files (signal = largest bin after
            # background removal; the crosstalk peak at the HOCAL position is much smaller than the signal)
            d = []
            for n in names[:12]:
                pth = os.path.join(a.data, n + ".csv")
                yy, _ = clean_file(pth, *dl_for(pth, (dl_z, dl_n)))
                d.append(int(np.argmax(yy)) - int(np.argmax(read_hocal(pth))))
            med = int(np.median(d)) if d else 0
            WIN_SHIFT = med if abs(med) > 2 else 0          # light pipe: signal sits on the HOCAL peak -> no shift
        print(f"--hocal: data window starts {WIN_SHIFT} bins after the HOCAL peak"
              + (" (set with --win-shift)" if a.win_shift is not None else " (found from the data)"))
    obs, Y = {}, {}
    for n in names:
        y, npul = clean_file(os.path.join(a.data, n + ".csv"), *dl_for(os.path.join(a.data, n + ".csv"), (dl_z, dl_n)))
        obs[n] = observables(y, time_ref(os.path.join(a.data, n + ".csv"), irf), npul) + (n_of(n),); Y[n] = y

    #references: from --ref-dir if given (stored under their own keys so they never mix with samples)
    def resolve(name, available, where):
        """Exact file name, or a unique file whose name starts with name + '_' (so 'B3' finds 'B3_2.0')."""
        if name in available:
            return name
        hits = [f for f in available if f.split("_")[0] == name]
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            ap.error(f"'{name}' matches several files in {where}: {', '.join(hits)} - give the full name")
        return None

    self_refs = [r for r in (a.self_ref or "").split(",") if r]
    ref_obs = obs
    if a.ref_dir is not None:
        ref_files = sorted(f[:-4] for f in os.listdir(a.ref_dir) if f.endswith(".csv"))
        rdl_z, rdl_n = read_hlong(a.ref_dl) if (a.ref_dl and BG_MODEL == "dl") else (dl_z, dl_n)
        if a.ref_irf and not a.hocal:
            ref_irf, _ = clean_file(os.path.join(a.ref_dir, a.ref_irf + ".csv"),
                                    *dl_for(os.path.join(a.ref_dir, a.ref_irf + ".csv"), (rdl_z, rdl_n)))
        else:
            ref_irf = irf
        ref_obs = {}
        # --refs must be in --ref-dir; --self-ref phantoms may be in --ref-dir or in --data
        for group, required in ((refs, True), (self_refs, False)):
            for i, r in enumerate(group):
                f = resolve(r, ref_files, a.ref_dir)
                if f is None:
                    if required:
                        ap.error(f"reference '{r}' not found in --ref-dir {a.ref_dir} (files there: {', '.join(ref_files)})")
                    f2 = resolve(r, list(obs), a.data)
                    if f2 is None:
                        ap.error(f"--self-ref '{r}' not found in --ref-dir or --data")
                    ref_obs[f2] = obs[f2]; group[i] = f2
                    continue
                y, npul = clean_file(os.path.join(a.ref_dir, f + ".csv"),
                                     *dl_for(os.path.join(a.ref_dir, f + ".csv"), (rdl_z, rdl_n)))
                ref_obs[f] = observables(y, time_ref(os.path.join(a.ref_dir, f + ".csv"), ref_irf), npul) + (n_of(f),)
                group[i] = f
        print(f"references {refs} read from {a.ref_dir}"
              f" (time zero: {'own #HOCAL' if a.hocal else (a.ref_irf + '.csv in ref-dir' if a.ref_irf else a.irf + '.csv from --data')},"
              f" DL: {a.ref_dl or a.dl})")
    elif not a.loo:
        for group, label in ((refs, "reference"), (self_refs, "--self-ref")):
            for i, r in enumerate(group):
                f = resolve(r, list(obs), a.data)
                if f is None:
                    ap.error(f"{label} '{r}' not found in --data {a.data} (add it there, or use --ref-dir)")
                group[i] = f
    if self_refs and not a.loo:
        print(f"self-reference: {refs} are evaluated with a calibration from {self_refs}")

    #load certified vals for reference
    truth = {}
    if a.truth:
        J = json.load(open(a.truth))
        truth = {k: (v["mean_mua"], v["mean_musp"]) for k, v in J.items()}
    # file names like "B2_2.0" use the certified values of "B2" (the part before the first "_")
    aliases = []
    for n in list(names) + refs + self_refs + list(ref_obs):
        base = n.split("_")[0]
        if n not in truth and base != n and base in truth:
            truth[n] = truth[base]; aliases.append(f"{n}->{base}")
    if aliases:
        print("certified values matched by name prefix: " + ", ".join(sorted(set(aliases))))
    if DL_USED:
        print("DL used for each file (minutes after the DL):")
        for f, (d, dt) in sorted(DL_USED.items()):
            flag = "   <- more than 3 h apart" if abs(dt) > 180 else ""
            print(f"   {f:28s} {d:24s} {dt:+8.0f} min{flag}")
    odd = {k: v[2] for k, v in list(obs.items()) + list(ref_obs.items()) if v[2] != N_TISSUE}
    print(f"refractive index: {N_TISSUE:g} for all samples" + (", except " + ", ".join(f"{k} n={v:g}" for k, v in
          sorted(odd.items())) + " (from the 'n' column)" if odd else ""))
    if comp:
        print(f"--compare: {len(comp)} phantoms in {os.path.basename(a.compare)}: {', '.join(comp)}")
        for n in list(names) + refs + self_refs + list(ref_obs):      # also usable as references / for error %
            k = match_name(n, comp)
            if k is not None and n not in truth:
                truth[n] = comp[k]
    if not a.loo:
        nocert = [r for r in refs + self_refs if r not in truth]
        if nocert:
            ap.error(f"no certified values for reference(s) {', '.join(nocert)} - add them to the --truth file")

    # fixed scattering: which musp each sample is held at (None = fitted free)
    FIXED, nofix = {}, []
    def musp_for(n):
        if a.fix_musp is None:
            return None
        if musp_const is not None:
            v = musp_const
        elif n in truth:
            v = truth[n][1]
        else:
            v = None; nofix.append(n)
        FIXED[n] = v
        return v
    if a.fix_musp is not None:
        print(f"--fix-musp: musp held at {f'{musp_const:g} cm-1' if musp_const is not None else 'each samples certified value'}"
              f", mua fitted from {dict(t='mean time only', I='intensity only', both='average of mean-time and intensity results')[MUSP_FIX_FROM]}"
              " (calibration on the references unchanged)")

    #calibrate and solve
    rows, calib_used = [], {}
    if a.loo:
        if a.ref_dir is not None:
            print("note: --loo ignores --ref-dir (it calibrates on the samples in --data)")
        known = [n for n in names if n in truth]
        for n in known:
            rho, t0, lk = calibrate(obs, truth, [m for m in known if m != n], None if a.fit_rho else a.rho)
            (mua, musp), err = invert(obs[n], rho, t0, lk, musp_for(n))
            rows.append((n, rho, t0, mua, err[0], musp, err[1])); calib_used[n] = (rho, t0, lk)
    else:
        fix = None if a.fit_rho else a.rho
        main_cal = calibrate(ref_obs, truth, refs, fix)
        alt_cal = calibrate(ref_obs, truth, self_refs, fix) if self_refs else None
        print(f"calibrated on {refs}: rho_eff = {main_cal[0]:.2f} mm, t0 = {main_cal[1]:.0f} ps")
        # when references sit in the data folder they are fitted as samples too (alt_cal if --self-ref);
        # with --ref-dir they are only used for calibration, and a check of each reference is printed instead
        for n in names:
            use_alt = alt_cal is not None and n in refs and a.ref_dir is None
            rho, t0, lk = alt_cal if use_alt else main_cal
            (mua, musp), err = invert(obs[n], rho, t0, lk, musp_for(n))
            rows.append((n, rho, t0, mua, err[0], musp, err[1])); calib_used[n] = (rho, t0, lk)
        if a.ref_dir is not None:
            chk = alt_cal if alt_cal is not None else main_cal
            print("reference check (each reference fitted with the "
                  + (f"--self-ref calibration from {self_refs}" if alt_cal is not None
                     else "calibration that includes itself, so it is circular; add --self-ref for an honest check")
                  + "):")
            for r in refs:
                (mua, musp), _ = invert(ref_obs[r], *chk, musp_for(r))
                c = truth[r]
                print(f"  {r:6s} mua {mua:.3f} ({100*(mua/c[0]-1):+.0f} %)  musp {musp:.2f} ({100*(musp/c[1]-1):+.0f} %)")

    # photon-noise (statistical) error of each averaged result, from the spread between blocks of the recording
    stat = {}
    if a.err_blocks:
        for n in calib_used:
            stat[n] = block_precision(os.path.join(a.data, n + ".csv"), irf,
                                      *dl_for(os.path.join(a.data, n + ".csv"), (dl_z, dl_n)), calib_used[n], a.err_blocks,
                                      n_idx=n_of(n), musp_fix=FIXED.get(n))

    if nofix:
        print(f"WARNING --fix-musp truth: no certified musp for {', '.join(sorted(set(nofix)))} - fitted with musp free")
    with open(a.out, "w") as fh:
        fh.write("sample,rho_eff_mm,t0_ps,mua_cm,mua_sd,musp_cm,musp_sd,mua_cert,musp_cert,mua_err_pct,musp_err_pct,"
                 "mua_sd_stat,musp_sd_stat,mua_sd_total,musp_sd_total,musp_fixed\n")
        with_noise = bool(a.err_blocks)
        print("sample  mua (cm-1)  ±prec" + (" ±noise" if with_noise else "") + "  | musp (cm-1) ±prec"
              + (" ±noise" if with_noise else "") + " | certified    | error %")
        for n, rho, t0, mua, sm, musp, ss in rows:
            c = truth.get(n, (np.nan, np.nan))
            e1, e2 = 100 * (mua / c[0] - 1), 100 * (musp / c[1] - 1)
            st = stat.get(n, (np.nan, np.nan))
            tot = [np.hypot(sm, st[0] if np.isfinite(st[0]) else 0), np.hypot(ss, st[1] if np.isfinite(st[1]) else 0)]
            fh.write(f"{n},{rho:.3f},{t0:.1f},{mua:.4f},{sm:.4f},{musp:.3f},{ss:.3f},{c[0]},{c[1]},{e1:.1f},{e2:.1f},"
                     f"{st[0]:.5f},{st[1]:.4f},{tot[0]:.4f},{tot[1]:.3f},{int(FIXED.get(n) is not None)}\n")
            noise_a = f" ±{st[0]:.4f}" if with_noise else ""
            noise_s = f" ±{st[1]:.3f}" if with_noise else ""
            known = f"| {c[0]:.3f} {c[1]:5.2f} | {e1:+4.0f} {e2:+4.0f}" if np.isfinite(c[0]) else "|   -     -   |   -    -"
            print(f"{n:6s}  {mua:.3f} ±{sm:.3f}{noise_a} | {musp:5.2f} " + ("fixed" if FIXED.get(n) is not None else
                  f"±{ss:4.2f}") + f"{noise_s} " + known)
    print("wrote", a.out, f"  (± precision from --sig-t {SIG_T:g} ps, --sig-i {SIG_I:g}"
          + (", ±noise = photon noise)" if with_noise else ")"))
    errs = np.array([[100 * (r[3] / truth[r[0]][0] - 1), 100 * (r[5] / truth[r[0]][1] - 1)]
                     for r in rows if r[0] in truth and (a.loo or r[0] not in refs)])
    if len(errs) >= 3:
        if a.fix_musp is not None:
            print("(musp is fixed, so its error below is the error of the fixed value, not a fit result)")
        print(f"accuracy vs certified over {len(errs)} phantoms: RMS error mua {np.sqrt(np.mean(errs[:,0]**2)):.0f} %, "
              f"musp {np.sqrt(np.mean(errs[:,1]**2)):.0f} %  (compare with the error bars)")

    fix_tag = "" if a.fix_musp is None else f" fixmusp-{a.fix_musp}-{MUSP_FIX_FROM}"
    if a.plot_linearity:
        plot_linearity(rows, truth, a.plot_dir, ("leave-one-out" if a.loo else "refs " + "+".join(refs)) + fix_tag,
                       stat, a.errbars)

    if comp:
        tag = ("leave-one-out" if a.loo else "refs " + "+".join(refs)) + fix_tag
        pts = plot_compare(rows, comp, os.path.join(a.plot_dir, f"compare_{tag}.png"), a.errbars,
                           () if a.loo else set(refs))
        if pts:
            sub = (f"time zero: {'HOCAL' if a.hocal else 'IRF'} · " + ("leave-one-out" if a.loo else
                   "refs " + ", ".join(refs)) + f" · ρ {a.rho:g} mm"
                   + ("" if a.fix_musp is None else f" · μs′ fixed ({a.fix_musp}), μa from {MUSP_FIX_FROM}"))
            plot_compare_tiles(pts, os.path.join(a.plot_dir, f"compare_tiles_{tag}.png"), sub)
            fn = os.path.join(os.path.dirname(a.out) or ".", "compare_results"
                              + ("" if a.fix_musp is None else f"_fixmusp-{a.fix_musp}-{MUSP_FIX_FROM}") + ".csv")
            with open(fn, "w") as fh:
                fh.write("sample,mua_expected,mua_recovered,mua_err_pct,musp_expected,musp_recovered,musp_err_pct\n")
                print("compare:  sample   mua exp -> rec  (err)    |  musp exp -> rec  (err)")
                for n, ea, ra, _, es, rs, _ in pts:
                    da, ds = 100 * (ra / ea - 1), 100 * (rs / es - 1)
                    fh.write(f"{n},{ea},{ra:.4f},{da:.1f},{es},{rs:.3f},{ds:.1f}\n")
                    note = ""
                    print(f"          {n:8s} {ea:.3f} -> {ra:.3f} ({da:+4.0f} %) | {es:5.2f} -> {rs:5.2f} ({ds:+4.0f} %){note}")
            print("wrote", fn)

    #frame-by-frame time series, using the same calibration as above (calibrated on the averaged references)
    if a.timeseries:
        if a.loo:
            print("--timeseries uses the --refs calibration; run it without --loo"); return
        ts_names = names if a.timeseries == "all" else [x for x in a.timeseries.split(",") if x]
        os.makedirs(a.ts_dir, exist_ok=True)
        all_ts = {}
        for n in ts_names:
            cal = alt_cal if (alt_cal is not None and n in refs and a.ref_dir is None) else main_cal
            rows_ts = fit_timeseries(os.path.join(a.data, n + ".csv"), irf,
                                     *dl_for(os.path.join(a.data, n + ".csv"), (dl_z, dl_n)), cal, a.group, n_idx=n_of(n),
                                     musp_fix=musp_for(n))
            if a.t_range:
                t_lo, t_hi = (float(v) for v in a.t_range.split("-"))
                rows_ts = [r for r in rows_ts if t_lo <= r[1] <= t_hi]
                if not rows_ts:
                    print(f"{n}: no time points inside --t-range {a.t_range}"); continue
            R = np.array(rows_ts, dtype=float)
            mua_s = moving_average(R[:, 4], a.smooth)        # equal to the raw mua when --smooth is 1
            fn = os.path.join(a.ts_dir, f"{n}_timeseries.csv")
            with open(fn, "w") as fh:
                w = max(a.smooth, 25)
                nm, ns = local_noise_sd(R[:, 4], w), local_noise_sd(R[:, 6], w)
                fh.write("frame,t_s,dT_ps,intensity,mua_cm,mua_sd,musp_cm,musp_sd,mua_smooth_cm,"
                         "mua_noise_sd,musp_noise_sd,mua_smooth_noise_sd\n")
                for r, ms, a1, a2 in zip(rows_ts, mua_s, nm, ns):
                    fh.write("%d,%.3f,%.2f,%.6g,%.5f,%.5f,%.4f,%.4f,%.5f" % (r + (ms,))
                             + ",%.5f,%.4f,%.5f\n" % (a1, a2, a1 / np.sqrt(a.smooth)))
            msg = (f"{n:6s} {len(R)} points over {R[-1,1]:.1f} s | mua {np.nanmean(R[:,4]):.3f} (SD {np.nanstd(R[:,4]):.4f},"
                   f" noise {noise_sd(R[:,4]):.4f}) | musp {np.nanmean(R[:,6]):.2f} (SD {np.nanstd(R[:,6]):.3f},"
                   f" noise {noise_sd(R[:,6]):.3f})")
            if a.smooth > 1:
                msg += f" | mua smoothed SD {np.nanstd(mua_s):.4f}"
            print(msg + f"  -> {fn}")
            all_ts[n] = rows_ts
            if not a.no_plot:
                plot_timeseries(rows_ts, n, os.path.join(a.ts_dir, f"{n}_timeseries.png"), truth.get(n), a.smooth,
                                a.ts_errors)
        if len(all_ts) > 1:
            os.makedirs(a.plot_dir, exist_ok=True)
            plot_timeseries_overview(all_ts, os.path.join(a.plot_dir, "timeseries_overview.png"), truth, a.relative, a.smooth)


if __name__ == "__main__":
    main()