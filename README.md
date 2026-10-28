# tmf-momenTOF

**TMF882x-Compatible Software for Time-Domain Diffuse Optics**

Real-time acquisition, calibration, and batch moments fitting ($\mu_a$, $\mu_s'$) for AMS OSRAM TMF882x direct Time-of-Flight (dToF) sensors based on the Contini '97 analytical semi-infinite model.

---

## Key Features

* **Live Acquisition & Fitting**: Connects to the AMS EVM GUI over TCP (`127.0.0.1:39998`) with automatic sensor detection (`TMF8828`, `TMF8821`, `TMF8820`) and real-time, frame-by-frame multi-zone Contini moments analysis.
* **Data Viewer**: Replay raw sensor recordings with adjustable speeds (1×–100×), switch channel overlays, inspect fitted time-series traces, and examine model residuals.
* **Offline Batch Analysis**: Automated high-throughput processing across multiple folders with publication-quality plot generation (PNG and interactive Plotly HTML) and linearity evaluation.
* **Validated Physics Engine**: Core moments solver (`tmf_moments_solver.py`) reproduces standard command-line fits with zero discrepancy.

---

## Installation & Quickstart

### Option 1: 1-Click Launch (Windows)
Double-click `run.bat`. On the first launch, it automatically installs all required dependencies before launching the application.

### Option 2: Manual Setup
```bash
pip install -r requirements.txt
python tmf_momenTOF.py
```

*Requirements: Python 3.9+, PySide6, pyqtgraph, numpy, scipy, matplotlib, plotly.*

---

## Workflow

### 1. Live Acquisition Tab
1. **Sensor & Acquisition (1)**: Connect to the AMS EVM GUI over TCP. The panel automatically detects the sensor model and active layout (3×3, 4×4, 3×6, or 8×8). Set iterations, SPAD map, and threshold. Advanced registers and on-chip calibration are available in the collapsible *Advanced Settings* drawer.
2. **Direct Light (DL) (2)**: Upload a Direct Light (`.csv`) recording (or record one in Section 3) to clean incoming photon histograms.
3. **Data Recording & Calibration (3)**: Choose what to record: **Sample Measurement**, **Reference Phantom (Calibration)**, or **Direct Light (DL)**.
   * Selecting *Reference Phantom* opens the calibration panel: set certified reference values ($\mu_a$, $\mu_s'$), probe separation $\rho$, and refractive indices ($n$, $n_\text{ext}$).
   * Click **Calibrate** to compute calibration factors via V12 and save a standard calibration `.json`.
4. **Real-Time Moments Fit (4)**: Load your calibration `.json` file, check **Enable real-time optical property fit (μa, μs′)**, and click **▶ Start Acquisition**. Saving a sample recording also exports fitted optical properties (`<name>_live_results.csv`).

### 2. Data Viewer Tab
* **Replay & Inspection**: Load any raw `.csv` sensor recording. Scrub through frames or run playback (1× to 100× speed). Toggle between single channel, summed channels, or multi-channel overlays.
* **Time-Series & Residuals**: Inspect recovered $\mu_a(t)$ and $\mu_s'(t)$ alongside fit residuals and ground-truth reference values.

### 3. Offline Batch Analysis Tab
* **Queue Data**: Click **Add files…** or **Add folder…** to load multiple recordings.
* **Configure Solver**: Select calibration `.json`, sample refractive indices, and fit mode (*Free fit*, *Fixed $\mu_s'$*, or *Fixed $\mu_a$*).
* **Run Analysis**: Click **Run analysis** to compute mean optical properties per file (`offline_results.csv`), linearity slopes (`linearity_fits.csv`), and interactive figures.

---

## Testing Without Hardware (Sensor Simulator)

You can test the entire live acquisition pipeline without a physical sensor board by launching the included mock server:

```bash
python tools/fake_evm_server.py test_files/sample.csv --loop
```

Then run `python tmf_momenTOF.py` and click **▶ Start Acquisition** on the Live tab.

---

## Repository Structure

| File / Folder | Description |
| :--- | :--- |
| `tmf_momenTOF.py` | Main application entry point (PySide6 + pyqtgraph) |
| `tmf_moments_solver.py` | Contini '97 analytical semi-infinite moments model solver |
| `core/` | Backend processing engines (`live_engine`, `offline_engine`, `calibrate_cli`, `figures`) |
| `gui/` | Modular UI package (`live_tab`, `viewer_tab`, `offline_tab`, `widgets`, `styles`, `persistence`) |
| `tools/fake_evm_server.py` | TCP hardware mock server for sensor simulation |
| `test_files/` | Sample raw EVM recording (`sample.csv`), calibration file, and expected values |
| `requirements.txt` | Python package dependencies |
| `run.bat` | 1-click Windows launcher script |
| `LICENSE` | MIT Open Source License |

---

## Citation

If you use this software or its methods in your research, please cite our paper:

> **Compact biomedical time-domain diffuse optical spectroscopy using a commercial LiDAR sensor**  
> Suraj Kumar Kothuri†, Keela Hughes†, Pranav Lanka, Claudia Nunzia Guadagno, Stefan Andersson-Engels, and Sanathana Konugolu Venkata Sekar. (*†These authors contributed equally*)

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
