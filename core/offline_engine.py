"""
offline_engine.py - fit recordings offline with a saved calibration (.json), using V12's own functions.

    python offline_engine.py job.json

job.json (written by the Offline tab):
{
  "calibration": "path/calibration.json",
  "files": ["path/a.csv", "path/b.csv", ...],           # from any number of folders
  "fit": {"mode": "both" | "fix_musp" | "fix_mua", "fix_value": 10.0, "fix_from": "both" | "t" | "I",
          "n": 1.41, "n_out": 1.0},
  "outputs": {"timeseries": true, "rolling_s": 0 (off) or seconds,
              "trim": null or [from_s, to_s], "compare": null or "known_values.csv"},
  "outdir": "path/results"
}

For every recording:
  - the AVERAGE result: all frames averaged, then fitted (as V12's moments_fit_results.csv)
  - optionally the TIME SERIES: every frame fitted (as V12 --timeseries; group 1)
Fits are identical to V12 for the free fit and fixed musp; fixed mua and an outside index other than 1 use the
same model as the Live tab.
"""
import datetime as _dt
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
for p in (ROOT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import live_engine as L        # noqa: E402  (patches the model for the outside index)
import tmf_moments_solver as V   # noqa: E402
import figures as FIG        # noqa: E402


def sample_names(files):
    """File stem; folder_stem when two files from different folders share a stem."""
    stems = [os.path.splitext(os.path.basename(f))[0] for f in files]
    out = []
    for f, s in zip(files, stems):
        out.append(f"{os.path.basename(os.path.dirname(f))}_{s}" if stems.count(s) > 1 else s)
    return out


def recalibrate(cfg, mode, irf_path, outdir, save_name="calibration_used.json", cal_dir=None, search_dirs=()):
    """Redo the calibration with the other time reference ('irf' or 'hocal'), from the reference recordings, DL and
    settings stored in the calibration. Returns the new calibration (also saved as calibration_used.json)."""
    import shutil, subprocess, tempfile
    refs = [dict(r) for r in (cfg.get("references") or [])]
    if not refs or not cfg.get("dl_file"):
        raise SystemExit("This calibration does not list its reference recordings / DL (made with an older version), "
                         "so it cannot be redone with another time zero - make a new one in the Calibrate tab.")
    s = cfg["settings"]
    work = tempfile.mkdtemp(prefix="ptof_recal_")
    try:
        truth, names = {}, []
        for r in refs:
            found = L.resolve_file(r["file"], r.get("file_rel"), cal_dir, search_dirs)
            if not found:
                raise SystemExit(f"reference recording not found: {r['file']} - put it in one of the data folders, "
                                 f"or next to the calibration file")
            if found != r["file"]:
                print(f"  reference {L._name(r['file'])} found at {found}")
            r["file"] = found
            name = os.path.splitext(L._name(found))[0]
            shutil.copyfile(found, os.path.join(work, name + ".csv"))
            truth[name] = {"mean_mua": r["mua"], "mean_musp": r["musp"]}; names.append(name)
        json.dump(truth, open(os.path.join(work, "_certified.json"), "w"))
        dl_file = L.resolve_file(cfg["dl_file"], cfg.get("dl_file_rel"), cal_dir, search_dirs)
        if not dl_file:
            raise SystemExit(f"DL not found: {cfg['dl_file']} - put it in one of the data folders (or its 'dl' "
                             f"sub-folder), or next to the calibration file")
        a = ["--data", work, "--dl", dl_file, "--truth", os.path.join(work, "_certified.json"),
             "--refs", ",".join(names), "--rho", f"{cfg['rho_mm']:g}", "--n", f"{s['N_TISSUE']:g}",
             "--bg-model", s["BG_MODEL"], "--bg-bins", ",".join(str(int(b)) for b in s["BG_IDX"]),
             "--ped-start", str(s["PED_START"]), "--win-pre", str(s["WIN_PRE"]), "--win-post", str(s["WIN_POST"]),
             "--sig-t", f"{s['SIG_T']:g}", "--sig-i", f"{s['SIG_I']:g}", "--min-signal", f"{s['MIN_SIGNAL']:g}",
             "--outdir", os.path.join(work, "out")]
        if s.get("ZONES"):
            a += ["--zones", ",".join(str(int(z[6:])) for z in s["ZONES"])]
        if s.get("N_OUT", 1.0) != 1.0:
            a += ["--n-out", f"{s['N_OUT']:g}"]
        if mode == "irf":
            stem = "irf_" + os.path.splitext(os.path.basename(irf_path))[0]
            shutil.copyfile(irf_path, os.path.join(work, stem + ".csv"))
            a += ["--irf", stem]
        else:
            a += ["--hocal"]
        new = os.path.join(outdir, save_name)
        print(f"re-doing the calibration with {'the IRF ' + os.path.basename(irf_path) if mode == 'irf' else 'HOCAL'}"
              f" as time zero, from {len(names)} reference recording(s)…", flush=True)
        p = subprocess.run([sys.executable, os.path.join(HERE, "calibrate_cli.py"), "--save-cal", new] + a,
                           capture_output=True, text=True)
        if p.returncode != 0 or not os.path.isfile(new):
            print(p.stdout[-3000:], p.stderr[-3000:])
            raise SystemExit("re-calibration failed")
        c = L.load_calibration(new)
        c["references"] = refs; c["irf_file"] = irf_path if mode == "irf" else None; c["dl_file"] = dl_file
        c["derived_from"] = "re-calibration with another time zero"
        json.dump(c, open(new, "w"), indent=1)
        print(f"  new t0 {c['t0_ps']:.1f} ps, logK {c['logk']:.4f} -> {save_name}")
        return c
    finally:
        shutil.rmtree(work, ignore_errors=True)


def make_ctx(cal_path, tz_mode, tz_irf, outdir, save_name, label, search_dirs=(), cfg=None, file_label=None,
             session_dl=None):
    """Everything needed to fit with one calibration: settings, IRF / DL histograms, t0, K, rho.
    cfg: a calibration given directly (the second reference stored inside cal_path)."""
    cfg = cfg or L.load_calibration(cal_path)
    mode = tz_mode or cfg["settings"]["TREF"]
    if mode == "irf" and cfg["settings"]["TREF"] == "hocal" and not tz_irf:
        raise SystemExit("IRF as time zero: choose the IRF recording")
    if mode != cfg["settings"]["TREF"]:
        cfg = recalibrate(cfg, mode, tz_irf, outdir, save_name, os.path.dirname(os.path.abspath(cal_path)), search_dirs)
    s = cfg["settings"]
    ctx = {"cfg": cfg, "s": s, "label": label, "file": file_label or os.path.basename(cal_path),
           "rho": cfg["rho_mm"], "t0": cfg["t0_ps"], "lk": cfg["logk"],
           "irf": None if cfg["irf"] is None else np.array(cfg["irf"]),
           "dl": None if cfg["dl"] is None else {k: np.array(v) for k, v in cfg["dl"].items()},
           "dl_n": cfg.get("dl_pulses")}
    if session_dl:                      # this session's DL for the data (V12 --dl); the references keep theirs
        z, n = V.read_hlong(session_dl)
        ctx["dl"], ctx["dl_n"] = {k: np.asarray(v, float) for k, v in z.items()}, n or ctx["dl_n"]
        print(f"background: DL {os.path.basename(session_dl)} for the data (calibration's DL for the references)")
    if s["TREF"] == "irf" and tz_irf and os.path.abspath(tz_irf) != os.path.abspath(cfg.get("irf_file") or ""):
        L.apply_settings(cfg)
        ctx["irf"], _ = V.clean_file(tz_irf, ctx["dl"], ctx["dl_n"])   # this session's IRF (V12 --irf / --ref-irf)
        print(f"time zero: IRF {os.path.basename(tz_irf)} for the data (calibration's IRF for the references)")
    print(f"{label} calibration {ctx['file']}: refs {', '.join(cfg.get('refs') or [])}, t0 {ctx['t0']:.1f} ps, "
          f"rho {ctx['rho']:.2f} mm, time zero {'HOCAL' if s['TREF'] == 'hocal' else 'IRF'}, background "
          f"{s['BG_MODEL']}", flush=True)
    return ctx


def main(job_path):
    job = json.load(open(job_path, encoding="utf-8"))
    fit = job["fit"]; out = job["outputs"]; outdir = job["outdir"]
    os.makedirs(outdir, exist_ok=True)
    tz = job.get("time_zero") or {}
    tz_mode, tz_irf = tz.get("mode"), tz.get("irf") or None
    search = list(dict.fromkeys(os.path.dirname(os.path.abspath(f)) for f in job["files"]))
    sdl = job.get("session_dl") or None
    main_ctx = make_ctx(job["calibration"], tz_mode, tz_irf, outdir, "calibration_used.json", "main", search,
                        session_dl=sdl)
    cfg = main_ctx["cfg"]
    refs = set(cfg.get("refs") or [])
    alt_ctx = None
    second = (json.load(open(job["calibration"], encoding="utf-8")).get("second_reference")
              if not job.get("ref_calibration") else None)
    if second:                          # stored in the calibration: fits the main reference when it is in the data
        alt_ctx = make_ctx(job["calibration"], tz_mode, tz_irf, outdir, "second_reference_used.json",
                           "second reference's", search, cfg=second,
                           file_label=f"second reference {', '.join(second.get('refs') or [])}", session_dl=sdl)
    elif job.get("ref_calibration"):
        alt_ctx = make_ctx(job["ref_calibration"], tz_mode, tz_irf, outdir, "ref_calibration_used.json", "references'",
                           search, session_dl=sdl)
        both = refs & set(alt_ctx["cfg"].get("refs") or [])
        if both:
            print(f"WARNING {', '.join(sorted(both))} is a reference of both calibrations - still circular")

    mode = fit.get("mode", "both")
    musp_fix = fit["fix_value"] if mode == "fix_musp" else None
    mua_fix = fit["fix_value"] if mode == "fix_mua" else None
    n_default = float(fit.get("n", cfg["settings"]["N_TISSUE"]))
    n_out = float(fit.get("n_out", cfg["settings"].get("N_OUT", 1.0)))
    if "win_pre" in fit:
        cfg.setdefault("settings", {})["WIN_PRE"] = int(fit["win_pre"])
    if "win_post" in fit:
        cfg.setdefault("settings", {})["WIN_POST"] = int(fit["win_post"])

    comp = {}
    if out.get("compare"):
        comp = V.read_compare(out["compare"])          # also fills V.NIDX from an 'n' column
        print(f"compare: {len(comp)} known phantoms in {os.path.basename(out['compare'])}")

    def n_of(name):
        k = V.match_name(name, V.NIDX)
        return V.NIDX[k] if k is not None else n_default

    def fitter_for(ctx, n):
        return L.LiveFitter(ctx["cfg"], 1, musp_fix, n, mua_fix=mua_fix, fix_from=fit.get("fix_from", "both"),
                            n_outside=n_out)

    def is_ref(name):
        return V.match_name(name, refs) is not None

    print(f"fit: {dict(both='mua and musp', fix_musp=f'mua (musp fixed at {musp_fix})', fix_mua=f'musp (mua fixed at {mua_fix})')[mode]}"
          f", sample index {n_default:g}, outside index {n_out:g}")

    files = job["files"]; names = sample_names(files)
    rows, all_ts = [], {}
    roll_s = float(out.get("rolling_s") or 0)
    trim = out.get("trim")
    ts_dir = os.path.join(outdir, "timeseries")
    for f, name in zip(files, names):
        ref_of_main = is_ref(name)
        ctx = alt_ctx if (ref_of_main and alt_ctx is not None) else main_ctx
        note = ("reference of the main calibration - fitted with " + ctx["file"]) if ctx is alt_ctx else \
               ("reference of the calibration - circular" if ref_of_main else "")
        print(f"\n{name}  ({f})" + (f"  [{note}]" if note else ""), flush=True)
        s, irf, dl, dl_n = ctx["s"], ctx["irf"], ctx["dl"], ctx["dl_n"]
        rho, t0, lk = ctx["rho"], ctx["t0"], ctx["lk"]
        n_idx = n_of(name)
        fitter = fitter_for(ctx, n_idx)       # applies that calibration's settings and the outside index
        # ---- average result (V12 clean_file + observables + invert)
        z, npul = V.read_hlong(f)
        npul = npul or dl_n
        if dl_n and npul != dl_n:
            print(f"  WARNING {npul // 1000}k iterations vs the calibration's DL {dl_n // 1000}k")
        if dl is None:
            y = sum(V.clean_zone(z[k], None, npul) for k in z if V.use_zone(k))
        else:
            y = sum(V.clean_zone(z[k], dl[k], npul) for k in z if k in dl and V.use_zone(k))
        ref = V.read_hocal(f) if s["TREF"] == "hocal" else irf
        o = tuple(V.observables(y, ref, npul)) + (n_idx,)
        if mua_fix is not None:
            m, sd = fitter._fit_musp_fixed_mua(o)
            x, err = (mua_fix, m), (0.0, sd)
        else:
            x, err = V.invert(o, rho, t0, lk, musp_fix)
        cal_label = ctx["file"] + (" (circular)" if (ref_of_main and ctx is main_ctx) else "")
        rows.append((name, f, float(x[0]), float(err[0]), float(x[1]), float(err[1]), n_idx, cal_label, rho, t0))
        print(f"  average: mua {x[0]:.4f} ± {err[0]:.4f}  musp {x[1]:.3f} ± {err[1]:.3f} cm-1")

        # ---- time series (V12 clean_frames + fit_timeseries, group 1)
        if out.get("timeseries"):
            Ys, T, npf = V.clean_frames(f, dl, dl_n, 1)
            H = V.clean_frames.hocal
            if s["TREF"] == "hocal":
                OBS = [tuple(V.observables(yy, h, npf)) + (n_idx,) for yy, h in zip(Ys, H)]
            else:
                OBS = [tuple(V.observables(yy, irf, npf)) + (n_idx,) for yy in Ys]
            I_all = np.array([ob[1] for ob in OBS], float)
            I_ref = np.nanmedian(I_all[I_all > 0]) if np.any(I_all > 0) else np.nan
            fitter = fitter_for(ctx, n_idx)
            R = []
            for i, (ob, t) in enumerate(zip(OBS, T)):
                r = fitter.fit_obs(ob, I_ref)
                R.append((i, float(t), float(ob[0]), float(ob[1])) + tuple(r))
            if trim:
                R = [r for r in R if trim[0] <= r[1] <= trim[1]]
            if not R:
                print("  no time points (check the time window)"); continue
            A = np.array(R, float)
            w = 1
            if roll_s > 0 and len(A) > 1:
                dt = np.nanmedian(np.diff(A[:, 1]))
                w = max(1, int(round(roll_s / dt))) if dt > 0 else 1
            ma, ms = V.moving_average(A[:, 4], w), V.moving_average(A[:, 6], w)
            os.makedirs(ts_dir, exist_ok=True)
            fn = os.path.join(ts_dir, f"{name}_timeseries.csv")
            with open(fn, "w") as fh:
                fh.write("frame,t_s,dT_ps,intensity,mua_cm,mua_sd,musp_cm,musp_sd,mua_rolling,musp_rolling\n")
                for r, a1, a2 in zip(R, ma, ms):
                    fh.write("%d,%.3f,%.2f,%.6g,%.5f,%.5f,%.4f,%.4f,%.5f,%.4f\n" % (r + (a1, a2)))
            blank = int(np.sum(~np.isfinite(A[:, 4])))
            print(f"  time series: {len(A)} points over {A[-1, 1] - A[0, 1]:.1f} s"
                  + (f", moving average {roll_s:g} s = {w} points" if w > 1 else "")
                  + (f", {blank} blank (low signal)" if blank else "") + f" -> {os.path.relpath(fn, outdir)}")
            all_ts[name] = R

    # ---- results table (+ known values)
    res = os.path.join(outdir, "offline_results.csv")
    with open(res, "w") as fh:
        fh.write("sample,file,mua_cm,mua_sd,musp_cm,musp_sd,fixed,n_sample,calibration,mua_expected,musp_expected,"
                 "mua_err_pct,musp_err_pct\n")
        for name, f, mua, sm, musp, ss, n_idx, cal_label, _, _ in rows:
            k = V.match_name(name, comp) if comp else None
            ea, es = comp[k] if k else (np.nan, np.nan)
            fh.write(f"{name},{f.replace(',', ' ')},{mua:.4f},{sm:.4f},{musp:.3f},{ss:.3f},"
                     f"{'mua' if mua_fix is not None else 'musp' if musp_fix is not None else ''},{n_idx:g},"
                     f"{cal_label.replace(',', ' ')},"
                     + (f"{ea:g},{es:g},{100 * (mua / ea - 1):.1f},{100 * (musp / es - 1):.1f}\n" if k else ",,,\n"))
    print(f"\nwrote {os.path.relpath(res, outdir)}")
    if comp:
        circular = {r[0] for r in rows if r[7].endswith("(circular)")}     # left out of the plots, as V12 does
        if circular:
            print(f"compare plots leave out {', '.join(sorted(circular))} (reference of the calibration: circular)")
        order = {k: i for i, k in enumerate(comp)}                         # in the order of the known-values CSV
        crow = sorted((order[V.match_name(r[0], comp)], V.natural_key(r[0]),
                       (r[0], r[2], r[3], r[4], r[5]) + tuple(comp[V.match_name(r[0], comp)][:2]))
                      for r in rows if V.match_name(r[0], comp) is not None and r[0] not in circular)
        if crow:                                   # plots are drawn (and downloaded) in the app's Plots tab
            FIG.write_fits([c[2] for c in crow], os.path.join(outdir, "linearity_fits.csv"))
    job["finished"] = _dt.datetime.now().isoformat(timespec="seconds")
    job["calibration_summary"] = {"t0_ps": main_ctx["t0"], "rho_mm": main_ctx["rho"], "logk": main_ctx["lk"],
                                  "refs": cfg.get("refs"), "time_zero": main_ctx["s"]["TREF"],
                                  "background": main_ctx["s"]["BG_MODEL"]}
    if alt_ctx:
        job["ref_calibration_summary"] = {"t0_ps": alt_ctx["t0"], "logk": alt_ctx["lk"],
                                          "refs": alt_ctx["cfg"].get("refs")}
    with open(os.path.join(outdir, "offline_job.json"), "w", encoding="utf-8") as fh:
        json.dump(job, fh, indent=1, ensure_ascii=False)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
