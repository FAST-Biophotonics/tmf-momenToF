# tmf-momenTOF

**TMF882x-Compatible Software for Time-Domain Diffuse Optics**

Real-time acquisition and moments analysis to recover optical properties (μa, μs′) from AMS OSRAM TMF882x direct Time-of-Flight sensors, based on the Contini '97 analytical semi-infinite model.

---

## Key Features

* **Live Acquisition & Analysis**: Connects to the AMS EVM GUI over TCP (`127.0.0.1:39998`) with automatic sensor detection (`TMF8828`, `TMF8821`, `TMF8820`) and real-time, frame-by-frame multi-zone Contini moments analysis.
* **Data Viewer**: Replay raw sensor recordings at adjustable speeds (0.25× to 10×, or max), switch between single-channel, summed, and overlaid channels, and compare fitted time series to expected values.
* **Offline Analysis**: Batch moments analysis across multiple files or folders, with optional figures (PNG and interactive Plotly HTML) showing the time-series variation of optical properties and comparison to expected values.

---

## Installation & Quickstart

### Option 1: 1-Click Launch (Windows)
Double-click `run.bat`. On the first launch, it automatically installs the required dependencies (listed in `requirements.txt`) before launching the application.

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
2. **Direct Light (DL) (2)**: Load a Direct Light (`.csv`) recording, or record one in Section 3. The DL recording provides the timing reference and is used to clean incoming photon histograms, mitigating the sensor's internal leak at bin 15.
3. **Data Recording & Calibration (3)**: Choose what to record: **Sample Measurement**, **Reference Phantom (Calibration)**, or **Direct Light (DL)**.
   * Selecting *Reference Phantom* opens the calibration panel: set the certified reference values (μa, μs′), source–detector separation ρ, and refractive indices (n, n_ext). An optional **Secondary Reference Phantom** can also be included.
   * Click **Calibrate** to compute the calibration factors (ρ, t₀, K) and save a calibration `.json`.
4. **Real-Time Moments Fit (4)**: Load your calibration `.json`, choose the fit mode (*μa and μs′ (free fit)*, *μa only (fix μs′)*, or *μs′ only (fix μa)*), check **Enable real-time optical property fit (μa, μs′)**, and click **▶ Start Acquisition**. Saving a sample recording also exports the fitted optical properties (`<name>_live_results.csv`).

### 2. Data Viewer Tab
* **Replay & Inspection**: Load any raw `.csv` sensor recording. Scrub through frames or play back at 0.25× to 10× (or max) speed. Show a single channel, the sum of selected channels, or an overlay of individual channels.
* **Time Series**: Inspect recovered μa(t) and μs′(t) alongside the expected (certified) values.

### 3. Offline Batch Analysis Tab
* **Queue Data**: Click **Add Files…** or **Add Folder…** to load multiple recordings.
* **Configure Solver**: Select the calibration `.json`, sample refractive indices, and fit mode (*μa and μs′ (free fit)*, *μa only (fix μs′)*, or *μs′ only (fix μa)*). Optionally set a *Session Direct Light (DL) Override*; if left blank, the DL stored in the calibration file is used.
* **Run Analysis**: Click **Run Analysis** to compute mean optical properties per file (`offline_results.csv`), linearity slopes (`linearity_fits.csv`), and interactive figures.

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
