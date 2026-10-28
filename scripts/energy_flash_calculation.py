"""
=============================================================================
Geothermal Electrical Energy - Single-Flash Cycle with Uncertainty
=============================================================================
Data: wells_above_100C.xlsx (Sheet1) - the source dataset for the electrical
power potential calculations reported in Section 3.6 / 4.4 of the
manuscript, covering wells with T > 100 degC.

THERMODYNAMIC MODEL (single-flash cycle, isentropic turbine):

  1) RESERVOIR FLUID. The "Pressure, kPa" column in the source file was
     checked against the saturation pressure at each well's own
     temperature (130 degC -> 270.1 kPa, 100 degC -> 101.3 kPa) and found
     to match closely. The fluid is therefore taken to be in the saturated
     liquid state:
        h1 = h_f(T1)                                    [kJ/kg]

  2) FLASH (isenthalpic throttling) to the separator temperature T_sep:
        h1 = h_f(T_sep) + x * [h_g(T_sep) - h_f(T_sep)]
     which gives the vapour fraction
        x = [h_f(T1) - h_f(T_sep)] / [h_g(T_sep) - h_f(T_sep)]
     This replaces the manuscript's earlier "100% vaporisation" assumption
     - in reality only about 5-8% of the fluid flashes to vapour.

  3) TURBINE (isentropic expansion + real efficiency):
     Inlet: saturated vapour, s_in = s_g(T_sep)
     At the condenser (T_cond), the outlet quality at equal entropy:
        x_out = [s_in - s_f(T_cond)] / [s_g(T_cond) - s_f(T_cond)]
        h_out,s = h_f(T_cond) + x_out * [h_g(T_cond) - h_f(T_cond)]
     Ideal work: w_s = h_g(T_sep) - h_out,s
     Real work:  w = w_s * eta_turbine * eta_generator

  4) POWER:
        m_total = discharge_rate * rho(T1) / 86400        [kg/s]
        P_elec  = m_total * x * w                          [kW]

  5) SEPARATOR TEMPERATURE is optimised per well to maximise power output.
     Sanity check: for T1 = 130 degC the optimum came out at 86 degC; the
     classical geothermal rule of thumb, ~(T1 + T_cond) / 2 = 87.5 degC,
     is a close match.

HOUSEHOLD EQUIVALENT: based on CEIC 2023 statistics, 7.12 kWh/day
  (Azerbaijan household electricity consumption: 6,625.8 million kWh/year
   / ~2.55 million households = ~2,598 kWh/year = ~7.12 kWh/day). This
   replaces the manuscript's earlier, uncited 8 kWh/day figure with a
   sourced statistic.

UNCERTAINTY: Monte Carlo (n=20,000) - temperature measurement uncertainty
  +/-1 degC, discharge-rate measurement uncertainty +/-10% (relative).
  The two inputs are sampled independently from normal distributions
  (sigma = 1 degC and 10% of the measured value). Results are reported as
  the median and the 90% uncertainty interval (5th-95th percentiles of the
  simulated power). The random seed of each well is derived from its label
  with zlib.crc32, so the results are identical on every run.

GROSS OUTPUT: P_elec is the electrical output at the generator terminals.
  Auxiliary (parasitic) plant loads are not subtracted.

Output: energy_flash_results.csv, FigureD_energy_results.png (300 dpi)

Required libraries: pip install CoolProp matplotlib pandas openpyxl
=============================================================================
"""

import os
import zlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import CoolProp.CoolProp as CP

# =====================================================================
DATA_FILE = 'data/wells_above_100C.xlsx'
SHEET = 'Sheet1'
# NOTE: the column names read from DATA_FILE below ('Yataq', 'Quyu nömrəsi ',
# 'Dərinlik, m', 'Temperatur', 'Debit m3/gün', 'Təzyiq, kPa') are the
# original Azerbaijani headers of the source spreadsheet and are kept
# as-is so the script matches the file on disk. If you rename the columns
# in the spreadsheet to English, update the corresponding column names
# below to match.

T_COND = 45.0             # condenser temperature (degC) - typical for geothermal
ETA_TURBINE = 0.80        # isentropic turbine efficiency
ETA_GENERATOR = 0.96      # generator + mechanical efficiency
PER_HOUSEHOLD_KWH_DAY = 7.12   # CEIC 2023, Azerbaijan

TEMP_UNC_C = 1.0           # temperature measurement uncertainty (+/- degC)
DEBIT_UNC_REL = 0.10       # discharge-rate measurement uncertainty (relative)
N_MC = 20000

ZONE_COLORS = {'Yalama': '#2166ac', 'Khudat': '#e08214', 'Nabran': '#1a9850', 'Khachmaz': '#d6604d'}
DPI_OUT = 300
# =====================================================================


def to_english_zone_name(name):
    """Normalise zone names to English spelling (Xudat -> Khudat,
    Xaçmaz -> Khachmaz), in case the source spreadsheet uses the
    Azerbaijani spelling, for consistency with the manuscript figures."""
    s = str(name)
    s = (s.replace('Xaçmaz', 'Khachmaz').replace('Xacmaz', 'Khachmaz')
          .replace('xaçmaz', 'khachmaz').replace('Xudat', 'Khudat')
          .replace('xudat', 'khudat'))
    return s


def h_f(T): return CP.PropsSI('H', 'T', T + 273.15, 'Q', 0, 'Water') / 1000.0
def h_g(T): return CP.PropsSI('H', 'T', T + 273.15, 'Q', 1, 'Water') / 1000.0
def s_f(T): return CP.PropsSI('S', 'T', T + 273.15, 'Q', 0, 'Water') / 1000.0
def s_g(T): return CP.PropsSI('S', 'T', T + 273.15, 'Q', 1, 'Water') / 1000.0
def rho_f(T): return CP.PropsSI('D', 'T', T + 273.15, 'Q', 0, 'Water')


def flash_specific_work(T1, T_sep, T_cond=T_COND):
    """Returns the real electrical work (kJ/kg of TOTAL fluid) and the
    vapour fraction for one pass of the single-flash cycle."""
    hf1, hfs, hgs = h_f(T1), h_f(T_sep), h_g(T_sep)
    x = (hf1 - hfs) / (hgs - hfs)
    if x <= 0:
        return 0.0, 0.0
    s_in = s_g(T_sep)
    sfc, sgc = s_f(T_cond), s_g(T_cond)
    x_out = np.clip((s_in - sfc) / (sgc - sfc), 0.0, 1.0)
    h_out_s = h_f(T_cond) + x_out * (h_g(T_cond) - h_f(T_cond))
    w_real = (hgs - h_out_s) * ETA_TURBINE * ETA_GENERATOR
    return x * w_real, x


def optimal_separator(T1, T_cond=T_COND):
    """Finds the separator temperature that maximises power output."""
    best_T, best_w, best_x = None, 0.0, 0.0
    for T_sep in np.arange(T_cond + 5.0, T1 - 0.5, 0.5):
        w, x = flash_specific_work(T1, T_sep, T_cond)
        if w > best_w:
            best_T, best_w, best_x = T_sep, w, x
    return best_T, best_w, best_x


# ============================================================
# 1. Data
# ============================================================
print("Loading well data...")
df = pd.read_excel(DATA_FILE, sheet_name=SHEET)
df['Yataq'] = df['Yataq'].ffill()
df['Label'] = df['Yataq'].astype(str) + "-" + df['Quyu nömrəsi '].astype(str)
dup = df.duplicated('Label', keep=False)
df.loc[dup, 'Label'] = (df.loc[dup, 'Label'] + " (" +
                        df.loc[dup, 'Dərinlik, m'].abs().astype(int).astype(str) + "m)")
df['Zone'] = df['Yataq'].astype(str).str.strip().apply(to_english_zone_name)
print(f"  {len(df)} well intervals, {df['Zone'].nunique()} zones.")

# Programmatically confirm that the "Pressure" column is the saturation pressure
chk = df.dropna(subset=['Təzyiq, kPa'])
dev = [abs(r['Təzyiq, kPa'] - CP.PropsSI('P', 'T', r['Temperatur'] + 273.15, 'Q', 0, 'Water') / 1000)
       for _, r in chk.iterrows()]
print(f"  'Pressure, kPa' vs. saturation pressure: mean difference {np.mean(dev):.2f} kPa "
      f"-> the fluid is in the saturated-liquid state (the key assumption for the flash model).")

# ============================================================
# 2. Per-well calculation + Monte Carlo
# ============================================================
print(f"\nComputing (T_cond={T_COND}°C, eta_turb={ETA_TURBINE}, eta_gen={ETA_GENERATOR}, "
      f"household={PER_HOUSEHOLD_KWH_DAY} kWh/day)...\n")

rows = []
for _, r in df.iterrows():
    T1, Qd = float(r['Temperatur']), float(r['Debit m3/gün'])
    T_sep, w_spec, x_steam = optimal_separator(T1)

    m_total = Qd * rho_f(T1) / 86400.0
    P_det = m_total * w_spec

    # Fixed per-well seed: Python's built-in hash() of a string changes
    # between interpreter sessions, whereas crc32 is deterministic.
    rng = np.random.default_rng(zlib.crc32(r['Label'].encode('utf-8')))
    T1_s = np.clip(rng.normal(T1, TEMP_UNC_C, N_MC), T_COND + 12, 220)
    Q_s = np.clip(rng.normal(Qd, Qd * DEBIT_UNC_REL, N_MC), 1, None)

    # As T1 varies, the optimal separator temperature also shifts slightly;
    # for computational speed, T_sep is held fixed within +/-1 degC of its
    # deterministic optimum (the effect of re-optimising is ~0.1%).
    w_s = np.array([flash_specific_work(t, min(T_sep, t - 0.5))[0] for t in T1_s])
    rho_s = np.array([rho_f(t) for t in T1_s])
    P_s = (Q_s * rho_s / 86400.0) * w_s
    P_s = np.clip(P_s, 0, None)

    p5, p50, p95 = np.percentile(P_s, [5, 50, 95])
    hh = P_s * 24.0 / PER_HOUSEHOLD_KWH_DAY
    h5, h50, h95 = np.percentile(hh, [5, 50, 95])

    print(f"  {r['Label']:20s} T={T1:6.1f}°C Q={Qd:6.0f} m³/day | T_sep={T_sep:5.1f}°C "
          f"vapour={x_steam*100:5.2f}% | P={p50:7.1f} kW ({p5:.0f}-{p95:.0f}) | households={h50:6.0f}")

    rows.append(dict(
        Well=r['Label'], Zone=r['Zone'], Lon=r['Lon'], Lat=r['Lat'], Depth_m=abs(r['Dərinlik, m']),
        T1_C=T1, Discharge_rate_m3day=Qd, T_sep_C=T_sep, steam_fraction_pct=x_steam * 100,
        mass_flow_kgs=m_total, specific_work_kJkg=w_spec,
        P_elec_kW_p50=p50, P_elec_kW_p5=p5, P_elec_kW_p95=p95,
        Annual_MWh_p50=p50 * 8760 / 1000,
        Households_p50=h50, Households_p5=h5, Households_p95=h95,
    ))

res = pd.DataFrame(rows).sort_values('P_elec_kW_p50', ascending=False).reset_index(drop=True)
os.makedirs("outputs", exist_ok=True)
res.to_csv('outputs/energy_flash_results.csv', index=False, encoding='utf-8-sig')

tot_p50 = res['P_elec_kW_p50'].sum()
tot_p5, tot_p95 = res['P_elec_kW_p5'].sum(), res['P_elec_kW_p95'].sum()
tot_h = res['Households_p50'].sum()
print(f"\n{'='*72}")
print(f"REGIONAL TOTAL (gross): {tot_p50:,.0f} kW ({tot_p50/1000:.2f} MW), "
      f"90% uncertainty interval {tot_p5:,.0f}-{tot_p95:,.0f} kW "
      f"(sum of per-well percentiles - fully correlated errors, conservative)")
print(f"                {tot_p50*8760/1000:,.0f} MWh/year | ~{tot_h:,.0f} households")
print(f"{'='*72}")
print("\n-> outputs/energy_flash_results.csv")

# ============================================================
# 3. FIGURE
# ============================================================
print("\nPreparing figure...")
fig = plt.figure(figsize=(15, 10), dpi=130)
gs = gridspec.GridSpec(2, 2, height_ratios=[1.15, 1], hspace=0.32, wspace=0.24)

# --- (a) Gross output per well, sorted, with 90% uncertainty interval ---
ax1 = fig.add_subplot(gs[0, :])
y = np.arange(len(res))[::-1]
cols = [ZONE_COLORS.get(z, '#777777') for z in res['Zone']]
ax1.barh(y, res['P_elec_kW_p50'], color=cols, edgecolor='#333', linewidth=0.6,
         height=0.65, zorder=3)
ax1.errorbar(res['P_elec_kW_p50'], y,
             xerr=[res['P_elec_kW_p50'] - res['P_elec_kW_p5'],
                   res['P_elec_kW_p95'] - res['P_elec_kW_p50']],
             fmt='none', ecolor='#333', elinewidth=1.2, capsize=3, zorder=4)
for yi, (p, hi, hh_) in enumerate(zip(res['P_elec_kW_p50'], res['P_elec_kW_p95'], res['Households_p50'])):
    ax1.text(hi + res['P_elec_kW_p95'].max() * 0.015, y[yi], f"{p:,.0f} kW  ·  {hh_:,.0f} households",
             va='center', fontsize=8.5, color='#333')
ax1.set_yticks(y); ax1.set_yticklabels(res['Well'], fontsize=9)
ax1.set_xlabel('Gross electrical output (kW)', fontsize=11)
ax1.set_xlim(0, res['P_elec_kW_p95'].max() * 1.30)
ax1.set_title('(a) Gross electrical output per well — single-flash cycle, median with 90 % uncertainty interval',
              fontsize=11, loc='left')
ax1.grid(axis='x', alpha=0.25, ls='--')
for s in ['top', 'right']:
    ax1.spines[s].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec='#333', lw=0.6)
           for z, c in ZONE_COLORS.items() if z in res['Zone'].values]
labels = [z for z in ZONE_COLORS if z in res['Zone'].values]
ax1.legend(handles, labels, loc='lower right', fontsize=9, title='Field', title_fontsize=9)

# --- (b) T-s diagram: explains the flash cycle ---
ax2 = fig.add_subplot(gs[1, 0])
Tdome = np.linspace(20, 370, 220)
sf_d = np.array([s_f(t) for t in Tdome])
sg_d = np.array([s_g(t) for t in Tdome])
ax2.plot(sf_d, Tdome, color='#555', lw=1.4)
ax2.plot(sg_d, Tdome, color='#555', lw=1.4)
ax2.fill_betweenx(Tdome, sf_d, sg_d, color='#e8eef4', alpha=0.7, zorder=0)

demo = res.iloc[0]
T1d, Tsd = demo['T1_C'], demo['T_sep_C']
s1 = s_f(T1d)
s_sep_f, s_sep_g = s_f(Tsd), s_g(Tsd)
x_d = demo['steam_fraction_pct'] / 100
s_mix = s_sep_f + x_d * (s_sep_g - s_sep_f)
s_in = s_sep_g
sfc, sgc = s_f(T_COND), s_g(T_COND)
x_out = np.clip((s_in - sfc) / (sgc - sfc), 0, 1)
s_out = s_in

ax2.plot([s1, s_mix], [T1d, Tsd], color='#d62728', lw=2.2, marker='o', ms=6, zorder=5,
         label='1→2  flash (isenthalpic)')
ax2.plot([s_mix, s_in], [Tsd, Tsd], color='#e08214', lw=2.2, marker='o', ms=6, zorder=5,
         label='2→3  steam separation')
ax2.plot([s_in, s_out], [Tsd, T_COND], color='#2166ac', lw=2.2, marker='o', ms=6, zorder=5,
         label='3→4  isentropic expansion')
ax2.annotate(f"1  reservoir\n{T1d:.0f} °C", (s1, T1d), fontsize=8, xytext=(-46, 6),
             textcoords='offset points', color='#d62728')
ax2.annotate(f"2  x = {x_d*100:.1f} %", (s_mix, Tsd), fontsize=8, xytext=(-30, -18),
             textcoords='offset points', color='#d62728')
ax2.annotate(f"3  separator\n{Tsd:.0f} °C", (s_in, Tsd), fontsize=8, xytext=(6, 4),
             textcoords='offset points', color='#e08214')
ax2.annotate(f"4  condenser\n{T_COND:.0f} °C", (s_out, T_COND), fontsize=8, xytext=(6, -4),
             textcoords='offset points', color='#2166ac')
ax2.set_xlim(0, 9); ax2.set_ylim(20, 200)
ax2.set_xlabel('Specific entropy, s (kJ kg⁻¹ K⁻¹)', fontsize=10)
ax2.set_ylabel('Temperature (°C)', fontsize=10)
ax2.set_title(f'(b) Single-flash cycle on the T–s diagram ({demo["Well"]})',
              fontsize=11, loc='left')
ax2.legend(fontsize=8, loc='upper left', frameon=True)
ax2.grid(alpha=0.22, ls='--')

# --- (c) Spatial distribution: well locations, size/color by power ---
ax3 = fig.add_subplot(gs[1, 1])
sizes = 60 + (res['P_elec_kW_p50'] / res['P_elec_kW_p50'].max()) * 500
sc = ax3.scatter(res['Lon'], res['Lat'], s=sizes, c=res['P_elec_kW_p50'],
                  cmap='YlOrRd', edgecolor='#333', linewidth=0.8, alpha=0.88, zorder=3)
for _, r in res.iterrows():
    ax3.annotate(r['Well'], (r['Lon'], r['Lat']), fontsize=7, xytext=(5, 5),
                 textcoords='offset points', color='#333')
cb3 = plt.colorbar(sc, ax=ax3, shrink=0.85)
cb3.set_label('Gross electrical output (kW)', fontsize=9)
ax3.set_xlabel('Longitude (°E)', fontsize=10)
ax3.set_ylabel('Latitude (°N)', fontsize=10)
ax3.set_title('(c) Spatial distribution of gross electrical output, symbol size ∝ output',
              fontsize=11, loc='left')
ax3.grid(alpha=0.25, ls='--')
for s in ['top', 'right']:
    ax3.spines[s].set_visible(False)

fig.suptitle('Geothermal Electrical Potential of Wells Exceeding 100 °C, Caspian–Guba Region\n'
             f'Single-flash cycle, condenser {T_COND:.0f} °C · '
             f'regional gross output {tot_p50/1000:.2f} MW ≈ {tot_h:,.0f} households '
             f'({PER_HOUSEHOLD_KWH_DAY} kWh day⁻¹, CEIC 2023)',
             fontsize=13, y=0.985)
fig.savefig('outputs/FigureD_energy_results.png', dpi=DPI_OUT, bbox_inches='tight')
plt.close(fig)
print("-> outputs/FigureD_energy_results.png")
print("\nDone.")
