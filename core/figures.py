"""
figures.py - every figure the Offline tab saves, in one consistent (paper) style:
sans-serif, open top/right, light horizontal grid, filled markers with a white edge, solid = recovered, dashed = expected,
μa blue, μs′ orange; no sample naming is assumed. The app's Plots tab draws the same plots interactively.
"""
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(HERE)
for p in (ROOT_DIR, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
from contextlib import contextmanager

import numpy as np

PAL = [("o", "#2f7bd6"), ("s", "#ec6a3d"), ("^", "#18a67a"), ("D", "#cc8a00"), ("v", "#8e5bd6"), ("P", "#d6457a"),
       ("X", "#0aa2c0"), ("*", "#6b6b6b"), ("h", "#7a8b1f"), ("p", "#b0542e")]
MUA_COL, MUSP_COL = PAL[0][1], PAL[1][1]
GRID, INK = "#e9e9e9", "#222222"
DPI = 200


@contextmanager
def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rc = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"], "font.size": 11,
          "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": INK, "axes.linewidth": 1.0,
          "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.9,
          "axes.axisbelow": True, "axes.titlelocation": "left", "axes.titlesize": 13, "axes.labelsize": 11.5,
          "xtick.direction": "out", "ytick.direction": "out", "xtick.color": INK, "ytick.color": INK,
          "legend.frameon": False, "legend.fontsize": 10, "lines.solid_capstyle": "round", "savefig.dpi": DPI,
          "savefig.facecolor": "white"}
    with plt.rc_context(rc):
        yield plt


def _pt(col, mk, ms=8):
    """filled marker with a white edge (as in the reference figure)"""
    return dict(marker=mk, ms=ms, mfc=col, mec="white", mew=1.2, color=col)


def _proxies(ax, extra=()):
    """legend entries: 'expected' dashed, 'recovered' solid (grey)"""
    ax.plot([], [], "--", color="#777777", lw=1.4, label="expected")
    ax.plot([], [], "-", color="#777777", lw=2.2, label="recovered")
    for e in extra:
        ax.plot([], [], **e)


def _save(fig, path, outdir=None, tight=True):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path); fig.clf()
    print(f"wrote {os.path.relpath(path, outdir) if outdir else path}")


MUA_LAB, MUSP_LAB = "μa (cm⁻¹)", "μs′ (cm⁻¹)"


# ----------------------------------------------------------------------------- data shared with the app's Plots tab
def rolling_window(t, window_s):
    if not window_s or len(t) < 2:
        return 1
    dt = np.nanmedian(np.diff(t))
    return max(1, int(round(window_s / dt))) if dt > 0 else 1


def overview_style(name, i):
    """marker / colour of the i-th recording when several are overlaid (cycles; no naming assumed)"""
    mk = PAL[(i // len(PAL)) % len(PAL)][0] if i >= len(PAL) else PAL[i][0]
    return mk, PAL[i % len(PAL)][1]


def linfit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 2:
        return np.nan, np.nan, np.nan
    a, b = np.polyfit(x, y, 1)
    ss = np.sum((y - np.mean(y)) ** 2)
    return a, b, (1 - np.sum((y - (a * x + b)) ** 2) / ss) if ss > 0 else np.nan


# ----------------------------------------------------------------------------- time series
def timeseries(R, name, path, expected=None, window=1, outdir=None, window_s=0):
    """One recording: mua (top) and musp (bottom) against time. Dots + line = every frame (pale when a moving
    average is drawn on top); dashed = expected value."""
    A = np.array(R, float)
    t = A[:, 1]
    with style() as plt:
        from tmf_moments_solver import moving_average
        fig, axs = plt.subplots(2, 1, figsize=(7.2, 5.6), sharex=True)
        for ax, j, lab, col, k in ((axs[0], 4, MUA_LAB, MUA_COL, 0), (axs[1], 6, MUSP_LAB, MUSP_COL, 1)):
            y = A[:, j]
            if window > 1:
                ax.plot(t, y, "-o", color=col, lw=0.8, ms=3, mec="none", alpha=0.3)
                ax.plot(t, moving_average(y, window), "-o", color=col, lw=2.2, ms=4, mec="white", mew=0.8, label="recovered (moving avg)")
            else:
                ax.plot(t, y, "-o", lw=1.4, label="recovered", **_pt(col, "o", 4.5))
            if expected is not None:
                ax.axhline(expected[k], color=col, ls="--", lw=1.4, alpha=0.9, label="expected")
            ax.set_ylabel(lab)
            if expected is not None or window > 1:
                ax.legend(loc="best")
        axs[1].set_xlabel("time (s)")
        axs[0].set_title(name + (f" ({window_s:g} s moving average)" if window_s else ""))
        _save(fig, path, outdir); plt.close(fig)


def overview(all_ts, path, window_s=0, outdir=None):
    """All recordings on one figure: mua (top), musp (bottom) against time; colour = series, marker = index."""
    with style() as plt:
        from tmf_moments_solver import moving_average
        fig, axs = plt.subplots(2, 1, figsize=(8.4, 6.0), sharex=True)
        fig.subplots_adjust(left=0.1, right=0.82, top=0.93, bottom=0.1, hspace=0.12)
        for i, (name, R) in enumerate(all_ts.items()):
            A = np.array(R, float); t = A[:, 1]
            mk, col = overview_style(name, i)
            w = rolling_window(t, window_s)
            for ax, j in ((axs[0], 4), (axs[1], 6)):
                y = moving_average(A[:, j], w) if w > 1 else A[:, j]
                ax.plot(t, y, "-", lw=1.6, label=name if ax is axs[0] else None, markevery=max(1, len(t) // 15),
                        **_pt(col, mk, 6))
        axs[0].set_ylabel(MUA_LAB); axs[1].set_ylabel(MUSP_LAB); axs[1].set_xlabel("time (s)")
        axs[0].set_title("all recordings" + (f" ({window_s:g} s moving average)" if window_s else ""))
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, loc="center left", bbox_to_anchor=(0.83, 0.5), ncol=1 if len(l) <= 16 else 2)
        _save(fig, path, outdir, tight=False); plt.close(fig)


# ----------------------------------------------------------------------------- compare
def nice_lim(top, step):
    return float(np.ceil(top * 1.08 / step) * step)


def all_linear(rows, errbars=False):
    """Recovered vs expected for every sample with known values. rows: [(name, mua, mua_sd, musp, musp_sd, mua_exp,
    musp_exp)]. Returns {'mua': (lim, fit (slope, intercept, r2), [(name, marker, colour, x, y, sd)]), 'musp': ...}"""
    out = {}
    for which, i_f, i_e, step in (("mua", 1, 5, 0.1), ("musp", 3, 6, 5.0)):
        pts = []
        col = MUA_COL if which == "mua" else MUSP_COL
        for r in rows:
            pts.append((r[0], "o", col, r[i_e], r[i_f], r[i_f + 1]))
        x = [p[3] for p in pts]; y = [p[4] for p in pts]
        top = max(max(x), max(np.add(y, [p[5] for p in pts]) if errbars else y))
        out[which] = (nice_lim(top, step), linfit(x, y), pts)
    return out


def _fit_text(fit, which="mua"):
    a, b, r2 = fit
    if not np.isfinite(a):
        return "fit: needs 2+ phantoms"
    return f"fit: slope {a:.2f}, intercept {b:+.3f}, R² {r2:.2f}" if which == "mua" else \
        f"fit: slope {a:.2f}, intercept {b:+.2f}, R² {r2:.2f}"


def compare(rows, path, outdir=None, errbars=False):
    """Every sample on one recovered-vs-expected plot (μa left, μs′ right): y = x dashed, ±10 % band shaded; every point labelled with its sample name."""
    if not rows:
        return
    data = all_linear(rows, errbars)
    with style() as plt:
        fig, axs = plt.subplots(1, 2, figsize=(11.6, 5.6))
        for ax, which, lab in ((axs[0], "mua", "μa"), (axs[1], "musp", "μs′")):
            lim, fit, pts = data[which]
            xx = np.array([0, lim])
            ax.fill_between(xx, 0.9 * xx, 1.1 * xx, color="#e9f5ee", lw=0, zorder=0, label="±10 %")
            ax.plot(xx, xx, "--", color="#777777", lw=1.4, label="y = x (expected)", zorder=1)
            for name, mk, col, x, y, sd in pts:
                ax.plot([x], [y], ls="none", zorder=4, **_pt(col, mk, 9))
                if errbars:
                    ax.errorbar([x], [y], yerr=[sd], fmt="none", ecolor=col, elinewidth=1.1, capsize=3, zorder=3)
                ax.annotate(name, (x, y), xytext=(5, 4), textcoords="offset points", fontsize=8.5, color="#444444")
            ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal"); ax.grid(True, axis="both")
            ax.set_xlabel(f"expected {lab} (cm⁻¹)"); ax.set_ylabel(f"recovered {lab} (cm⁻¹)")
            ax.set_title(f"{lab}: recovered vs expected"); ax.legend(loc="upper left", fontsize=9)
        _save(fig, path, outdir); plt.close(fig)


# ----------------------------------------------------------------------------- linearity
def write_fits(rows, path):
    """linearity_fits.csv: recovered = slope x expected + intercept over all samples, for μa and μs′.
    rows as for compare()."""
    with open(path, "w") as fh:
        fh.write("quantity,n,slope,intercept,r2\n")
        for which, i_f, i_e in (("mua", 1, 5), ("musp", 3, 6)):
            a, b, r2 = linfit([r[i_e] for r in rows], [r[i_f] for r in rows])
            fh.write(f"{which},{len(rows)},{a:.4f},{b:.4f},{r2:.4f}\n")
            print(f"linearity {'mua ' if which == 'mua' else 'musp'}: recovered = {a:.3f} x expected {b:+.4f}, "
                  f"R² {r2:.3f}  ({len(rows)} samples)")


# ----------------------------------------------------------------------------- interactive HTML (plotly)
_PL_SYM = {"o": "circle", "s": "square", "^": "triangle-up", "D": "diamond", "v": "triangle-down", "P": "cross",
           "X": "x", "*": "star", "h": "hexagon", "p": "pentagon"}


def _pl_fig(rows=1, titles=None, shared_x=True, height=None):
    from plotly.subplots import make_subplots
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=shared_x, vertical_spacing=0.12 if titles else 0.06,
                        subplot_titles=titles)
    fig.update_layout(template="simple_white", font=dict(family="Arial, Helvetica, sans-serif", size=13, color=INK),
                      height=height or (420 if rows == 1 else 680), width=None, margin=dict(l=70, r=30, t=60, b=60),
                      legend=dict(bgcolor="rgba(0,0,0,0)"), hovermode="closest")
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, ticks="outside")
    fig.update_xaxes(ticks="outside", showgrid=False)
    for a in fig.layout.annotations:          # subplot titles on the left, like the PNGs
        a.update(x=0, xanchor="left", font=dict(size=15))
    return fig


def _mk(col, mk, size=10):
    return dict(symbol=_PL_SYM.get(mk, "circle"), size=size, color=col, line=dict(color="white", width=1.2))


def _err(sd, on, col):
    return dict(type="data", array=list(sd), visible=True, color=col, thickness=1.2, width=4) if on else None


def _write_html(fig, path, title, plotlyjs=True):
    fig.update_layout(title=dict(text=title, x=0.01, xanchor="left", font=dict(size=17)) if title else None)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.write_html(path, include_plotlyjs=plotlyjs, full_html=True,
                   config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 2}})
    print(f"wrote {path}")


def html_timeseries(all_ts, path, window_s=0, expected=None, plotlyjs=True):
    """one recording (μa / μs′ coloured, expected dashed) or several overlaid (colour = series, marker = index)"""
    import plotly.graph_objects as go
    from tmf_moments_solver import moving_average
    fig = _pl_fig(2)
    single = len(all_ts) == 1
    for i, (name, R) in enumerate(all_ts.items()):
        A = np.array(R, float); t = A[:, 1]; w = rolling_window(t, window_s)
        for row, j, col0, k in ((1, 4, MUA_COL, 0), (2, 6, MUSP_COL, 1)):
            y = A[:, j]
            if single:
                if w > 1:
                    fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", line=dict(color=col0, width=0.8),
                                             marker=dict(size=4, color=col0), opacity=0.3, name="every frame",
                                             showlegend=False), row=row, col=1)
                    fig.add_trace(go.Scatter(x=t, y=moving_average(y, w), mode="lines", line=dict(color=col0, width=2.6),
                                             name=f"recovered ({window_s:g} s moving avg)", showlegend=False), row=row, col=1)
                else:
                    fig.add_trace(go.Scatter(x=t, y=y, mode="lines+markers", line=dict(color=col0, width=1.4),
                                             marker=_mk(col0, "o", 6), name="recovered", showlegend=False), row=row, col=1)
                if expected is not None:
                    fig.add_hline(y=expected[k], line=dict(color=col0, dash="dash", width=1.6), row=row, col=1)
            else:
                mk, col = overview_style(name, i)
                fig.add_trace(go.Scatter(x=t, y=moving_average(y, w) if w > 1 else y, mode="lines+markers",
                                         line=dict(color=col, width=1.6), marker=_mk(col, mk, 7), name=name,
                                         legendgroup=name, showlegend=row == 1), row=row, col=1)
    fig.update_yaxes(title_text=MUA_LAB, row=1, col=1); fig.update_yaxes(title_text=MUSP_LAB, row=2, col=1)
    fig.update_xaxes(title_text="time (s)", row=2, col=1)
    title = (next(iter(all_ts)) + (f" ({window_s:g} s moving average)" if window_s else "")) if single else f"{len(all_ts)} recordings" + (f" ({window_s:g} s moving average)" if window_s else "")
    _write_html(fig, path, title, plotlyjs)


def html_compare(rows, path, errbars=False, plotlyjs=True):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    data = all_linear(rows, errbars)
    fig = make_subplots(rows=1, cols=2, subplot_titles=["μa: recovered vs expected", "μs′: recovered vs expected"], horizontal_spacing=0.1)
    fig.update_layout(template="simple_white", font=dict(family="Arial, Helvetica, sans-serif", size=13, color=INK),
                      height=600, margin=dict(l=70, r=30, t=60, b=60), legend=dict(bgcolor="rgba(0,0,0,0)"))
    for a in fig.layout.annotations:
        a.update(font=dict(size=15))
    for c, which, lab in ((1, "mua", "μa"), (2, "musp", "μs′")):
        lim, fit, pts = data[which]
        xx = [0, lim]
        fig.add_trace(go.Scatter(x=xx + xx[::-1], y=[0, 1.1 * lim, 0.9 * lim, 0], fill="toself", fillcolor="#e9f5ee",
                                 mode="lines", line=dict(width=0), name="±10 %", legendgroup="band", showlegend=c == 1,
                                 hoverinfo="skip"), row=1, col=c)
        fig.add_trace(go.Scatter(x=xx, y=xx, mode="lines", line=dict(color="#777777", dash="dash", width=1.6),
                                 name="y = x (expected)", legendgroup="yx", showlegend=c == 1), row=1, col=c)
        for name, mk, col, x, y, sd in pts:
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[name], textposition="top right",
                                     textfont=dict(size=10, color="#444444"), marker=_mk(col, mk, 11), name=name,
                                     legendgroup=name, showlegend=False, error_y=_err([sd], errbars, col),
                                     hovertemplate=f"{name}<br>expected %{{x:.3g}}<br>recovered %{{y:.3g}}"
                                                   f"<br>error {100 * (y / x - 1):+.1f} %<extra></extra>"), row=1, col=c)
        fig.update_xaxes(title_text=f"expected {lab} (cm⁻¹)", range=[0, lim], ticks="outside", showgrid=True,
                         gridcolor=GRID, constrain="domain", row=1, col=c)
        fig.update_yaxes(title_text=f"recovered {lab} (cm⁻¹)", range=[0, lim], ticks="outside", showgrid=True,
                         gridcolor=GRID, scaleanchor="x" if c == 1 else "x2", scaleratio=1, constrain="domain", row=1, col=c)
    _write_html(fig, path, "", plotlyjs)




# ----------------------------------------------------------------------------- any app plot (Live, Calibrate)
# panels: [{"title", "xlabel", "ylabel", "logx", "logy", "hlines": [(y, colour, dashed)],
#           "curves": [{"x", "y", "name", "colour", "width", "dashed", "line", "marker", "size"}]}]
_MPL_FROM_PG = {"o": "o", "s": "s", "t": "v", "t1": "^", "d": "D", "+": "P", "x": "X", "star": "*", "h": "h", "p": "p"}


def generic_png(panels, path, title=""):
    with style() as plt:
        n = len(panels)
        fig, axs = plt.subplots(n, 1, figsize=(7.4, 3.0 + 2.6 * n), sharex=n > 1, squeeze=False)
        for ax, pn in zip(axs[:, 0], panels):
            for c in pn["curves"]:
                kw = dict(color=c["colour"], lw=c["width"] if c["line"] else 0, ls="--" if c["dashed"] else "-",
                          label=c["name"] or None)
                if c["marker"]:
                    kw.update(marker=_MPL_FROM_PG.get(c["marker"], "o"), ms=c["size"], mfc=c["colour"], mec="white",
                              mew=0.8 if c["size"] < 6 else 1.2)
                if not c["line"]:
                    kw["ls"] = "none"
                ax.plot(c["x"], c["y"], **kw)
            for y, col, dashed in pn.get("hlines", []):
                ax.axhline(y, color=col, ls="--" if dashed else "-", lw=1.4)
            if pn.get("logy"):
                ax.set_yscale("log")
            if pn.get("logx"):
                ax.set_xscale("log")
            if pn.get("xrange"):
                ax.set_xlim(*pn["xrange"])
            if pn.get("yrange"):
                ax.set_ylim(*pn["yrange"])
            ax.set_ylabel(pn.get("ylabel", "")); ax.set_title(pn.get("title", ""))
            if any(c["name"] for c in pn["curves"]):
                ax.legend(loc="best")
        axs[-1, 0].set_xlabel(panels[-1].get("xlabel", ""))
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=13)
        _save(fig, path); plt.close(fig)


def generic_html(panels, path, title="", plotlyjs=True):
    import plotly.graph_objects as go
    fig = _pl_fig(len(panels), [p.get("title", "") for p in panels] if any(p.get("title") for p in panels) else None,
                  height=420 if len(panels) == 1 else 340 * len(panels))
    for r, pn in enumerate(panels, start=1):
        for c in pn["curves"]:
            mode = "+".join([m for m, on in (("lines", c["line"]), ("markers", bool(c["marker"]))) if on]) or "markers"
            fig.add_trace(go.Scatter(x=list(c["x"]), y=list(c["y"]), mode=mode, name=c["name"] or "",
                                     showlegend=bool(c["name"]),
                                     line=dict(color=c["colour"], width=c["width"], dash="dash" if c["dashed"] else None),
                                     marker=_mk(c["colour"], _MPL_FROM_PG.get(c["marker"], "o"), c["size"] + 2)),
                          row=r, col=1)
        for y, col, dashed in pn.get("hlines", []):
            fig.add_hline(y=y, line=dict(color=col, dash="dash" if dashed else None, width=1.6), row=r, col=1)
        fig.update_yaxes(title_text=pn.get("ylabel", ""), type="log" if pn.get("logy") else None, row=r, col=1)
        fig.update_xaxes(type="log" if pn.get("logx") else None, row=r, col=1)
        if pn.get("xrange"):
            fig.update_xaxes(range=list(pn["xrange"]), row=r, col=1)
        if pn.get("yrange"):
            rng = pn["yrange"]
            fig.update_yaxes(range=[np.log10(v) for v in rng] if pn.get("logy") else list(rng), row=r, col=1)
    fig.update_xaxes(title_text=panels[-1].get("xlabel", ""), row=len(panels), col=1)
    _write_html(fig, path, title, plotlyjs)
