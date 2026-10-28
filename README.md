# Caspian-Guba Geothermal Assessment

Geostatistical and thermodynamic assessment of the geothermal energy
potential of the Caspian–Guba region, Azerbaijan. This repository contains
the complete, reproducible analysis pipeline behind the manuscript:

> Islamzade, A.V., Mammadov, P.Y. *Uncertainty-Quantified Geostatistical
> Modeling and Thermodynamic Power Assessment of Geothermal Reservoirs in
> the Caspian-Guba Region, Azerbaijan.* Geoenergy Science and Engineering
> (submitted).

## What this is (and is not)

The regional temperature, hydraulic conductivity, and discharge-rate
fields are estimated by **geostatistical interpolation (kriging)** of
well-test data, not by solving a physics-based numerical (PDE) flow/heat
model. Electrical power potential is computed from a **physically
rigorous single-flash thermodynamic cycle** using real IAPWS-97 water/steam
properties, with Monte Carlo uncertainty propagation. Every estimated
field is validated by formal Leave-One-Out Cross-Validation (LOOCV), and
the validation metrics (RMSE, MAE, R²) are reported alongside the results.

## Pipeline overview

Run the scripts in this order. Each stage's output feeds the next.

| Step | Script | Input | Output |
|---|---|---|---|
| 1 | `scripts/temperature_model_trend_kriging.py` | `data/temperature_observations.xlsx`, `data/well_names.xlsx`, `data/discharge_rate_intervals.xlsx`, `data/caspian_guba_boundary.shp` | `outputs/3d_geothermal_temperature_model_trend_kriging.html`, `outputs/temperature_trend_kriging_model.npz`, `outputs/validation_report_temperature_trend.txt` |
| 2 | `scripts/discharge_rate_model_kriging.py` | `data/discharge_rate_intervals.xlsx`, `data/well_names.xlsx`, `data/caspian_guba_boundary.shp` | `outputs/3d_discharge_rate_model_kriging.html`, `outputs/discharge_rate_kriging_model.npz`, `outputs/validation_report_discharge_rate.txt` |
| 3 | `scripts/conductivity_figures.py` | `data/hydraulic_conductivity_tests.xlsx` | `outputs/FigureA_conductivity_overview.png`, `outputs/FigureB_conductivity_transect.png`, `outputs/FigureC_correlation_structure.png`, `outputs/conductivity_zonal_statistics.txt` |
| 4 | `scripts/energy_flash_calculation.py` | `data/wells_above_100C.xlsx` | `outputs/energy_flash_results.csv`, `outputs/FigureD_energy_results.png` |
| 5 | `scripts/prepare_energy_summary.py` | `outputs/energy_flash_results.csv` (from step 4) | `outputs/energy_map_data.csv` |
| 6 | `scripts/electrical_potential_figure.py` | `outputs/energy_map_data.csv` (from step 5) | `outputs/Figure_geothermal_electrical_potential.png` |
| 7 | `scripts/energy_map.py` | `outputs/energy_map_data.csv` (from step 5) | `outputs/Figure6_energy_map_corrected.png` |

Steps 1–3 (temperature, discharge rate, hydraulic conductivity) are
independent of one another and can be run in any order. Steps 4–7 must run
in sequence, since each depends on the previous step's output.

```bash
python scripts/temperature_model_trend_kriging.py
python scripts/discharge_rate_model_kriging.py
python scripts/conductivity_figures.py
python scripts/energy_flash_calculation.py
python scripts/prepare_energy_summary.py
python scripts/electrical_potential_figure.py
python scripts/energy_map.py
```

Run every script from the **repository root** (not from inside `scripts/`),
so the relative `data/` and `outputs/` paths resolve correctly.

## Directory structure

```
.
├── data/            # input spreadsheets and boundary shapefile (see below) - not tracked in this repo, see "Data" section
├── scripts/          # the seven pipeline scripts described above
├── outputs/          # created automatically; all figures, HTML models, and result tables are written here
├── requirements.txt
└── README.md
```

## Data

Because the underlying well and pumping-test data include coordinates and
identifiers for specific properties, the raw spreadsheets are not
distributed in this public repository. Researchers who wish to reproduce
the analysis with the original dataset, or who need clarification on the
expected column layout of each input file, are welcome to contact the
corresponding author.

Each script's expected input file and columns are documented in a comment
block near the top of the script (the `# --- USER SETTINGS ---` /
configuration section). At minimum, the following files are required in
`data/`:

| File | Required columns |
|---|---|
| `temperature_observations.xlsx` | `Longitude`, `Latitude`, `Depth (m)`, `Temperature (°C)` |
| `discharge_rate_intervals.xlsx` | `Lon`, `Lat`, `from_depth_m`, `to_depth_m`, `Discharge_rate` |
| `hydraulic_conductivity_tests.xlsx` (sheet `Sheet1`) | `Quyu ad` (well name), `Lon`, `Lat`, `From Depth (m)`, `To Depth (m)`, `Keçiricilik (K)`, `Q - Debit, m3/gün`, `Temperatur, C`, `M -sulu kompleksin qalınlığı`, `S=S(statik)-S(dinamik)` |
| `wells_above_100C.xlsx` (sheet `Sheet1`) | `Yataq` (field/zone), `Quyu nömrəsi ` (well number), `Lat`, `Lon`, `Dərinlik, m`, `Temperatur`, `Debit m3/gün`, `Təzyiq, kPa` |
| `well_names.xlsx` | `Well_Name`, `Lon`, `Lat` |
| `caspian_guba_boundary.shp` (+ `.dbf`/`.shx`/`.prj`) | study-area boundary polygon, any CRS (scripts reproject automatically to EPSG:32639) |

Note: `hydraulic_conductivity_tests.xlsx` and `wells_above_100C.xlsx`
still use their original Azerbaijani column headers, since the scripts
that read them (`conductivity_figures.py`, `energy_flash_calculation.py`)
must match the actual file on disk. If you rename these columns to
English in your own copy of the spreadsheets, update the corresponding
column names in those two scripts to match.

## Methodology summary

- **Temperature** — Universal Kriging (linear depth trend + local kriging
  of the residuals), selected between Spherical and Exponential variogram
  models by Leave-One-Out Cross-Validation.
- **Hydraulic conductivity** — computed per pumping test via the Logan
  (1964) approximation, characterised by cross-validated zonal statistics
  (median, geometric mean, range) rather than a continuous surface, since
  LOOCV showed that a continuous kriging surface does not outperform
  simple zonal statistics at the available well density.
- **Discharge rate** — Ordinary Kriging on log₁₀-transformed values
  (local, moving-window), validated by LOOCV.
- **Electrical power potential** — a complete single-flash thermodynamic
  cycle (isenthalpic flash to a per-well-optimised separator temperature,
  isentropic turbine expansion, real turbine/generator efficiencies),
  using IAPWS-97 water/steam properties via the CoolProp library.
  Uncertainty is propagated by Monte Carlo simulation (n = 20,000) from
  ±1 °C temperature and ±10% discharge-rate measurement uncertainty;
  results are reported as medians with 90% uncertainty intervals
  (5th–95th percentiles). Values are gross electrical output at the
  generator terminals; auxiliary plant loads are not subtracted.
  The Monte Carlo seed of each well is fixed (zlib.crc32 of the well
  label), so repeated runs give identical results. Per-well values may
  differ from Table 3 of the manuscript by up to ±1 kW (median) and
  ±2 kW (interval bounds), owing to Monte Carlo sampling; the regional
  total (2.32 MW) is unchanged.

Full derivations, equations, and references are given in the manuscript
(Sections 3.2–3.6).

## Requirements

```bash
pip install -r requirements.txt
```

Note: `CoolProp` (thermodynamic properties) and `geopandas`/`cartopy`
(spatial/mapping) can occasionally require platform-specific installation
steps; see their respective documentation if `pip install` fails on your
system.

## License

MIT License — see `LICENSE`.

## Citation

If you use this code, please cite the manuscript above (full citation to
be updated upon publication).
