"""
=============================================================================
Prepare Energy Summary - bridges energy_flash_calculation.py to the figure
scripts (electrical_potential_figure.py, energy_map.py)
=============================================================================
energy_flash_calculation.py performs the full per-well single-flash +
Monte Carlo calculation and writes a detailed results table
(outputs/energy_flash_results.csv, with mass flow, vapour fraction,
separator temperature, etc.).

electrical_potential_figure.py and energy_map.py only need a compact
summary (well name, coordinates, median power, 90% uncertainty interval, household
equivalent). This script performs that column selection/renaming so the
two figure scripts have a single, clearly named input file to read,
instead of each guessing at the detailed table's column names.

Run this after energy_flash_calculation.py and before
electrical_potential_figure.py / energy_map.py.

Input:  outputs/energy_flash_results.csv   (from energy_flash_calculation.py)
Output: outputs/energy_map_data.csv        (consumed by the two figure scripts)
=============================================================================
"""

import os
import pandas as pd

IN_FILE = 'outputs/energy_flash_results.csv'
OUT_FILE = 'outputs/energy_map_data.csv'

print(f"Loading {IN_FILE} ...")
df = pd.read_csv(IN_FILE)

summary = df.rename(columns={
    'P_elec_kW_p50': 'Energy_kWh',
    'P_elec_kW_p5': 'Energy_CI_low_kWh',
    'P_elec_kW_p95': 'Energy_CI_high_kWh',
    'Households_p50': 'Household_count',
})[['Well', 'Lat', 'Lon', 'Energy_kWh', 'Energy_CI_low_kWh', 'Energy_CI_high_kWh', 'Household_count']]

os.makedirs('outputs', exist_ok=True)
summary.to_csv(OUT_FILE, index=False)
print(f"-> {OUT_FILE} ({len(summary)} wells)")
print("\nYou can now run electrical_potential_figure.py and energy_map.py.")
