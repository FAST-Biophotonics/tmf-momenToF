"""
calibrate_cli.py - run the V12 moments fit on a reference session and save the calibration for live fitting.

    python calibrate_cli.py --save-cal my_cal.json  <any ptof_moments_v12.py arguments>

Everything V12 normally does still happens (results CSV, plots, reference check printed). In addition, a JSON file is
written holding everything the live fitter needs, so a live session does not need the original files:
    - the calibration (rho, t0, logK) found by V12's own calibrate()
    - V12's settings after its argument parsing (background model and bins, window, HOCAL shift, n, sigmas, zones ...)
    - the cleaned IRF histogram (not with --hocal) and the DL histograms of every zone (with the dl background)
"""
import argparse
import datetime as _dt
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
for p in (ROOT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import tmf_moments_solver as V  # noqa: E402


def _pick_dl(spec, data_dir):
    """Same rules as V12 for --dl: a file, a folder of DL files, or a comma-separated list. For the live session one
    DL is needed: with several, the most recently recorded one is used (and reported)."""
    if not spec:
        return None
    paths = []
    parts = [spec] if os.path.exists(spec) else [x.strip() for x in spec.split(",") if x.strip()]
    for part in parts:
        if os.path.isdir(part):
            csvs = sorted(f for f in os.listdir(part) if f.lower().endswith(".csv"))
            if data_dir and os.path.abspath(part) == os.path.abspath(data_dir):
                csvs = [f for f in csvs if f.lower().startswith("dl")]
            paths += [os.path.join(part, f) for f in csvs]
        elif os.path.isfile(part):
            paths.append(part)
    if not paths:
        return None
    if len(paths) > 1:
        timed = [(V.start_time(p) or 0.0, p) for p in paths]
        chosen = max(timed)[1]
        print(f"[calibration] several DL files - the live session will use the most recent: {os.path.basename(chosen)}")
        return chosen
    return paths[0]


def main():
    argv = sys.argv[1:]
    if "--save-cal" not in argv:
        raise SystemExit("usage: calibrate_cli.py --save-cal FILE.json [--n-out N] <ptof_moments_v12 arguments>")
    i = argv.index("--save-cal")
    save = argv[i + 1]
    v12_args = argv[:i] + argv[i + 2:]
    n_out = 1.0
    if "--n-out" in v12_args:                       # outside index: not a V12 option, handled here
        j = v12_args.index("--n-out"); n_out = float(v12_args[j + 1]); v12_args = v12_args[:j] + v12_args[j + 2:]
    if n_out != 1.0:
        import live_engine                          # replaces V12's model with the one that has an outside index
        live_engine.set_outside_index(n_out)
    if "--loo" in v12_args:
        raise SystemExit("--loo has no single calibration - remove it to save a calibration for live fitting")

    # the few arguments needed to rebuild the live inputs (V12 parses everything itself)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--data"); ap.add_argument("--dl"); ap.add_argument("--irf", default="irf_10mm")
    ap.add_argument("--hocal", action="store_true"); ap.add_argument("--refs", default="B2,B3,C2,C3,D2")
    ap.add_argument("--rho", type=float, default=18.0); ap.add_argument("--outdir", "--out-dir", dest="outdir")
    ap.add_argument("--out", default="moments_fit_results.csv"); ap.add_argument("--fix-musp", default=None)
    a, _ = ap.parse_known_args(v12_args)

    captured = []
    original = V.calibrate

    def capture(obs, truth, refs, rho=18.0):
        r = original(obs, truth, refs, rho)
        captured.append({"refs": list(refs), "cal": [float(x) for x in r],
                         "truth": {k: [float(truth[k][0]), float(truth[k][1])] for k in refs}})
        return r

    V.calibrate = capture
    sys.argv = ["tmf_moments_solver.py"] + v12_args
    V.main()
    if not captured:
        raise SystemExit("[calibration] Moments solver did not calibrate - nothing saved")
    main_cal = captured[0]

    dl_path = _pick_dl(a.dl, a.data) if V.BG_MODEL == "dl" else None
    dl_z, dl_n = (V.read_hlong(dl_path) if dl_path else (None, None))
    irf = None
    irf_path = None
    if V.TREF != "hocal":
        irf_path = os.path.join(a.data, a.irf + ".csv")
        irf, _ = V.clean_file(irf_path, dl_z, dl_n)

    out_csv = os.path.join(a.outdir, a.out) if a.outdir else a.out
    ref_rows = []
    if os.path.isfile(out_csv):
        with open(out_csv) as fh:
            head = fh.readline().strip().split(",")
            for line in fh:
                ref_rows.append(dict(zip(head, line.strip().split(","))))

    cfg = {
        "format": "ptof-moments-live-calibration/1",
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "engine": "tmf_moments_solver",
        "v12_args": v12_args,
        "refs": main_cal["refs"],
        "ref_truth": main_cal["truth"],
        "rho_mm": main_cal["cal"][0], "t0_ps": main_cal["cal"][1], "logk": main_cal["cal"][2],
        "settings": {
            "BG_IDX": [int(x) for x in V.BG_IDX], "BG_MODEL": V.BG_MODEL, "PED_START": int(V.PED_START),
            "WIN_PRE": int(V.WIN_PRE), "WIN_POST": int(V.WIN_POST), "WIN_SHIFT": int(V.WIN_SHIFT),
            "TREF": V.TREF, "N_TISSUE": float(V.N_TISSUE), "SIG_T": float(V.SIG_T), "SIG_I": float(V.SIG_I),
            "MIN_SIGNAL": float(V.MIN_SIGNAL), "ZONES": sorted(V.ZONES) if V.ZONES else None,
            "MUSP_FIX_FROM": V.MUSP_FIX_FROM, "N_OUT": n_out, "DT_PS": float(V.DT_PS), "FRAME_PERIOD_S": float(V.FRAME_PERIOD_S),
        },
        "irf_file": irf_path, "irf": None if irf is None else [float(x) for x in irf],
        "dl_file": dl_path, "dl_pulses": dl_n,
        "dl": None if dl_z is None else {k: [float(x) for x in v] for k, v in dl_z.items()},
        "session_results": ref_rows,
    }
    os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
    with open(save, "w") as fh:
        json.dump(cfg, fh, indent=1)
    print(f"[calibration] rho {cfg['rho_mm']:.2f} mm, t0 {cfg['t0_ps']:.1f} ps, logK {cfg['logk']:.4f} "
          f"(refs {', '.join(cfg['refs'])}) - saved to {save}")


if __name__ == "__main__":
    main()
