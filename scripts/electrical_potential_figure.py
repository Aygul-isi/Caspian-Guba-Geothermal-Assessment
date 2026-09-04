"""
=============================================================================
Geothermal Electrical Potential of Wells Exceeding 100 C - Figure
=============================================================================
Reproduces the 3-panel figure:
  (a) Net electrical power per well (single-flash cycle), horizontal bar
      chart with 90% confidence interval, colored by field, sorted
      descending, with "kW / households" labels.
  (b) T-s (temperature-entropy) diagram of the single-flash cycle for the
      flagship well (Yalama-1): 1 reservoir -> 2 flash (isenthalpic) ->
      3 separator -> 4 condenser (isentropic expansion). Saturation dome
      and state points computed from real IAPWS water properties via
      CoolProp (not approximated), so the flash quality at the separator
      comes out at x = 8.1% for a 130 C reservoir / 86 C separator, exactly
      matching the original figure.
  (c) Spatial distribution of electrical power (marker size and color both
      proportional to power).

Data source: energy_map_data.csv (Well, Lat, Lon, Energy_kWh,
Energy_CI_low_kWh, Energy_CI_high_kWh, Household_count) - this is the same
underlying dataset used for the energy contour map (energy_map.py); here
"Energy_kWh" is read as the single-flash net electrical power in kW.

Household equivalent: households = power_kW * 24 / 7.12
(7.12 kWh/household/day, CEIC 2023 residential consumption figure).

Required libraries: pip install pandas numpy matplotlib CoolProp
=============================================================================
"""

import numpy as np
import pandas as pd
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from CoolProp.CoolProp import PropsSI

# =====================================================================
DATA_FILE = 'outputs/energy_map_data.csv'
RESERVOIR_T_C = 130.0     # state 1: reservoir liquid temperature (Yalama-1)
SEPARATOR_T_C = 86.0      # state 3: separator saturation temperature
CONDENSER_T_C = 45.0      # state 4: condenser temperature
KWH_PER_HOUSEHOLD_DAY = 7.12   # CEIC 2023
FIELD_COLORS = {'Yalama': '#2166ac', 'Khudat': '#e08214', 'Khachmaz': '#d6604d'}
FIELD_ORDER = ['Yalama', 'Khudat', 'Khachmaz']
DPI_OUT = 300
# =====================================================================


def get_field(well_name):
    n = well_name.lower()
    if 'yalama' in n:
        return 'Yalama'
    if 'khudat' in n:
        return 'Khudat'
    if 'khachmaz' in n:
        return 'Khachmaz'
    return 'Other'


def clean_display_name(raw_well):
    """Well names arriving from prepare_energy_summary.py are already
    unique and well-formatted (e.g. 'Yalama-17 (3990m)', produced by
    energy_flash_calculation.py's own duplicate-handling logic). This
    function only trims stray whitespace and normalises any remaining
    underscore-style names (e.g. a manually-edited CSV using 'Yalama _1')
    to the hyphenated form used throughout the manuscript figures."""
    return raw_well.strip().replace(' _', '-').replace('_', '-')


# =====================================================================
# LOAD DATA
# =====================================================================
print("Loading well power data...")
df = pd.read_csv(DATA_FILE)
df.columns = df.columns.str.strip()
os.makedirs('outputs', exist_ok=True)

df['DisplayName'] = df['Well'].apply(clean_display_name)
df['Field'] = df['DisplayName'].apply(get_field)

df = df.rename(columns={
    'Energy_kWh': 'Power_kW',
    'Energy_CI_low_kWh': 'CI_low',
    'Energy_CI_high_kWh': 'CI_high',
})
df['Households'] = (df['Power_kW'] * 24.0 / KWH_PER_HOUSEHOLD_DAY).round()

total_power_MW = df['Power_kW'].sum() / 1000.0
total_households = int(df['Households'].sum())
print(f"  {len(df)} wells exceeding 100 C, regional total "
      f"{total_power_MW:.2f} MW ~ {total_households:,} households.")

df_sorted = df.sort_values('Power_kW', ascending=True).reset_index(drop=True)

# =====================================================================
# FIGURE LAYOUT
# =====================================================================
fig = plt.figure(figsize=(14, 13), dpi=130)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.35, 1], hspace=0.32, wspace=0.28)

# ---------------------------------------------------------------------
# (a) Electrical power per well - horizontal bar chart with 90% CI
# ---------------------------------------------------------------------
axA = fig.add_subplot(gs[0, :])
y = np.arange(len(df_sorted))
colors = df_sorted['Field'].map(FIELD_COLORS)
xerr = np.vstack([
    df_sorted['Power_kW'] - df_sorted['CI_low'],
    df_sorted['CI_high'] - df_sorted['Power_kW'],
])
axA.barh(y, df_sorted['Power_kW'], color=colors, edgecolor='none', height=0.62, zorder=3)
axA.errorbar(df_sorted['Power_kW'], y, xerr=xerr, fmt='none', ecolor='black',
             elinewidth=1.1, capsize=4, capthick=1.1, zorder=4)

for yi, r in zip(y, df_sorted.itertuples()):
    label = f"{r.Power_kW:.0f} kW \u00b7 {r.Households:,.0f} households"
    axA.text(r.CI_high + 12, yi, label, va='center', fontsize=9.5, color='#333')

axA.set_yticks(y)
axA.set_yticklabels(df_sorted['DisplayName'], fontsize=9.5)
axA.set_xlabel('Net electrical power (kW)', fontsize=11)
axA.set_xlim(0, df_sorted['CI_high'].max() * 1.4)
axA.set_title('(a) Electrical power per well \u2014 single-flash cycle, median with 90% confidence interval',
              fontsize=11.5, loc='left')
axA.grid(axis='x', alpha=0.25, ls='--')
for s_ in ['top', 'right']:
    axA.spines[s_].set_visible(False)

legend_handles = [plt.Rectangle((0, 0), 1, 1, color=FIELD_COLORS[f]) for f in FIELD_ORDER]
axA.legend(legend_handles, FIELD_ORDER, title='Field', loc='lower right',
           frameon=True, fontsize=9.5, title_fontsize=9.5)

# ---------------------------------------------------------------------
# (b) T-s diagram of the single-flash cycle (Yalama-1), real water
#     properties via CoolProp
# ---------------------------------------------------------------------
axB = fig.add_subplot(gs[1, 0])


def sat_hs(T_C, Q):
    T_K = T_C + 273.15
    h = PropsSI('H', 'T', T_K, 'Q', Q, 'Water') / 1000.0   # kJ/kg
    s = PropsSI('S', 'T', T_K, 'Q', Q, 'Water') / 1000.0   # kJ/kg-K
    return h, s


# saturation dome, 1 C to 200 C
T_dome = np.linspace(1.0, 200.0, 300)
s_liq = np.array([sat_hs(t, 0)[1] for t in T_dome])
s_vap = np.array([sat_hs(t, 1)[1] for t in T_dome])
axB.fill_betweenx(T_dome, s_liq, s_vap, color='#eaf1f8', zorder=1)
axB.plot(s_liq, T_dome, color='#444', lw=1.3, zorder=2)
axB.plot(s_vap, T_dome, color='#444', lw=1.3, zorder=2)

# state points
h1, s1 = sat_hs(RESERVOIR_T_C, 0)                      # 1: reservoir liquid
hf3, sf3 = sat_hs(SEPARATOR_T_C, 0)
hg3, sg3 = sat_hs(SEPARATOR_T_C, 1)                     # 3: separator (sat. vapor)
x2 = (h1 - hf3) / (hg3 - hf3)                           # flash quality (isenthalpic)
s2 = sf3 + x2 * (sg3 - sf3)                             # 2: mixture at separator T
hf4, sf4 = sat_hs(CONDENSER_T_C, 0)
hg4, sg4 = sat_hs(CONDENSER_T_C, 1)
x4 = (sg3 - sf4) / (sg4 - sf4)                          # 4: isentropic expansion 3->4

axB.plot([s1, s2], [RESERVOIR_T_C, SEPARATOR_T_C], color='#d62728', marker='o',
          lw=2, zorder=5, label='1\u21922  flash (isenthalpic)')
axB.plot([s2, sg3], [SEPARATOR_T_C, SEPARATOR_T_C], color='#e08214', marker='o',
          lw=2, zorder=5, label='2\u21923  steam separation')
axB.plot([sg3, sg3], [SEPARATOR_T_C, CONDENSER_T_C], color='#1f77b4', marker='o',
          lw=2, zorder=5, label='3\u21924  isentropic expansion')

axB.annotate(f'1  reservoir\n{RESERVOIR_T_C:.0f} \u00b0C', (s1, RESERVOIR_T_C),
             xytext=(-8, 12), textcoords='offset points', color='#d62728', fontsize=9.5)
axB.annotate(f'2  x = {x2*100:.1f} %', (s2, SEPARATOR_T_C),
             xytext=(-8, -20), textcoords='offset points', color='#d62728', fontsize=9.5)
axB.annotate(f'3  separator\n{SEPARATOR_T_C:.0f} \u00b0C', (sg3, SEPARATOR_T_C),
             xytext=(6, 8), textcoords='offset points', color='#e08214', fontsize=9.5)
axB.annotate(f'4  condenser\n{CONDENSER_T_C:.0f} \u00b0C', (sg3, CONDENSER_T_C),
             xytext=(6, -6), textcoords='offset points', color='#1f77b4', fontsize=9.5)

axB.set_xlim(0, 9); axB.set_ylim(20, 200)
axB.set_xlabel('Specific entropy, s (kJ kg\u207b\u00b9 K\u207b\u00b9)', fontsize=10.5)
axB.set_ylabel('Temperature (\u00b0C)', fontsize=10.5)
axB.set_title('(b) Single-flash cycle on the T\u2013s diagram (Yalama-1)', fontsize=11.5, loc='left')
axB.legend(loc='upper left', fontsize=8.5, frameon=True)
axB.grid(alpha=0.2, ls='--')

# ---------------------------------------------------------------------
# (c) Spatial distribution of electrical power
# ---------------------------------------------------------------------
axC = fig.add_subplot(gs[1, 1])
sizes = 40 + (df['Power_kW'] / df['Power_kW'].max()) * 500
scC = axC.scatter(df['Lon'], df['Lat'], s=sizes, c=df['Power_kW'], cmap='YlOrRd',
                   edgecolor='#333', linewidth=0.7, alpha=0.9, zorder=3)
for _, r in df.iterrows():
    axC.annotate(r['DisplayName'], (r['Lon'], r['Lat']), fontsize=7.5,
                 xytext=(5, 4), textcoords='offset points', color='#333')
cbar = plt.colorbar(scC, ax=axC, shrink=0.85, pad=0.02)
cbar.set_label('Electrical power (kW)', fontsize=10, rotation=270, labelpad=16)
axC.set_xlabel('Longitude (\u00b0E)', fontsize=10.5)
axC.set_ylabel('Latitude (\u00b0N)', fontsize=10.5)
axC.set_title('(c) Spatial distribution of electrical power, symbol size \u221d power',
              fontsize=11.5, loc='left')
axC.grid(alpha=0.25, ls='--')

fig.suptitle(
    'Geothermal Electrical Potential of Wells Exceeding 100 \u00b0C, Caspian\u2013Guba Region\n'
    f'Single-flash cycle, condenser {CONDENSER_T_C:.0f} \u00b0C \u00b7 regional total '
    f'{total_power_MW:.2f} MW \u2248 {total_households:,} households '
    f'({KWH_PER_HOUSEHOLD_DAY:.2f} kWh day\u207b\u00b9, CEIC 2023)',
    fontsize=14.5, y=0.995)

fig.savefig('outputs/Figure_geothermal_electrical_potential.png', dpi=DPI_OUT, bbox_inches='tight')
plt.close(fig)
print("-> outputs/Figure_geothermal_electrical_potential.png")
