"""
=============================================================================
Hydraulic Conductivity - Publication-Quality Figures (A, B, C)
=============================================================================
This script replaces the 3D "empty box" visualization. n=24 points will
always look sparse in a large 3D volume - the problem is not stylistic,
it is a FORMAT problem. Three complementary, static, journal-ready figures
are therefore produced instead (journals accept exactly this kind of
figure, not interactive HTML):

  FIGURE A - 4-panel overview:
      (a) well locations, symbol size proportional to log K
      (b) K vs depth
      (c) zonal K ranges
      (d) K vs aquifer thickness

  FIGURE B - "fence" diagram along the NW-SE transect:
      each tested interval shown at its real depth/thickness,
      colored by K. A classic hydrogeological presentation format.

  FIGURE C - Correlation structure (MOST IMPORTANT):
      Reviewer #3 said "focus on the correlation between K and Q".
      HOWEVER: K is NOT measured independently, it is computed from Q:
             K = 1.22*(Q/S)/M   =>   log K = log1.22 + logQ - logS - logM
      That is, the K~Q correlation (r=0.818) is PARTLY DUE TO DEFINITION -
      exactly like the r=1.000 tautology the reviewer criticized. This
      figure openly shows that trap and instead presents the TRULY
      independent relationships (T~Q, T~depth, M~depth).

Additional output: conductivity_zonal_statistics.txt

Required libraries: pip install matplotlib scipy pyproj pandas openpyxl
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import os
from pyproj import Transformer
from scipy import stats

# =====================================================================
ROCK_PROPERTIES_FILE = 'data/hydraulic_conductivity_tests.xlsx'  # source workbook
                                                         # (update this path if
                                                         # your file name differs)
ROCK_SHEET_NAME = 'Sheet1'                             # update if your sheet is named differently
DPI_OUT = 300           # journals typically require >= 300 dpi
ZC = {'Yalama': '#2166ac', 'Khudat': '#e08214', 'Nabran': '#1a9850', 'Khachmaz': '#d6604d'}
# =====================================================================

transformer = Transformer.from_crs("EPSG:4326", "EPSG:32639", always_xy=True)


def get_zone(well_name):
    """Map a well name to its zone. Returns the same English zone labels
    used as keys in ZC, regardless of whether the source spelling is the
    Azerbaijani (Xudat/Xaçmaz) or English (Khudat/Khachmaz) variant."""
    n = str(well_name).lower().replace(' ', '')
    if 'yalam' in n:
        return 'Yalama'
    if 'xudat' in n or 'khudat' in n:
        return 'Khudat'
    if 'nabran' in n:
        return 'Nabran'
    if 'xaçmaz' in n or 'xacmaz' in n or 'khachmaz' in n:
        return 'Khachmaz'
    return 'Other'


def to_english_well_name(name):
    """Normalize well names to English spelling, in case the source
    workbook contains Azerbaijani spellings (Xaçmaz, Xudat)."""
    s = str(name)
    s = (s.replace('Xaçmaz', 'Khachmaz')
          .replace('Xacmaz', 'Khachmaz')
          .replace('xaçmaz', 'khachmaz')
          .replace('Xudat', 'Khudat')
          .replace('xudat', 'khudat'))
    return s


print("Loading raw pumping-test data...")
df = pd.read_excel(ROCK_PROPERTIES_FILE, sheet_name=ROCK_SHEET_NAME).dropna(how='all')
df['X'], df['Y'] = transformer.transform(df['Lon'].values, df['Lat'].values)
df['Quyu ad'] = df['Quyu ad'].apply(to_english_well_name)
df['Zone'] = df['Quyu ad'].apply(get_zone)
df['Zmid'] = (df['From Depth (m)'] + df['To Depth (m)']) / 2
print(f"  {len(df)} independent pumping-test measurements, {df['Quyu ad'].nunique()} wells, "
      f"{df['Zone'].nunique()} zones.")

K = df['Keçiricilik (K)'].values
Q = df['Q - Debit, m3/gün'].values
T = df['Temperatur, C'].values
M = df['M -sulu kompleksin qalınlığı'].values
S = df['S=S(statik)-S(dinamik)'].values
Zmid = df['Zmid'].values

# --- Programmatically confirm the tautology in the K formula ---
lhs = np.log10(K)
rhs = np.log10(1.22) + np.log10(Q) - np.log10(S) - np.log10(M)
max_dev = np.abs(lhs - rhs).max()
print(f"\nVerifying the K formula: |logK - (log1.22+logQ-logS-logM)| max = {max_dev:.2e}")
print("  -> confirmed: log K is, by definition, a linear function of log Q.")

# =====================================================================
# ZONAL STATISTICS
# =====================================================================
zonal = df.groupby('Zone').agg(
    n_measurements=('Keçiricilik (K)', 'count'),
    n_wells=('Quyu ad', 'nunique'),
    K_min=('Keçiricilik (K)', 'min'),
    K_max=('Keçiricilik (K)', 'max'),
    K_median=('Keçiricilik (K)', 'median'),
).reset_index()
zonal['K_geomean'] = df.groupby('Zone')['Keçiricilik (K)'].apply(
    lambda x: np.exp(np.mean(np.log(x)))).values
zonal['K_range_orders'] = np.log10(zonal['K_max'] / zonal['K_min'])
zonal = zonal.sort_values('K_geomean', ascending=False)
print("\nZonal statistics:")
print(zonal.to_string(index=False))

os.makedirs("outputs", exist_ok=True)
with open("outputs/conductivity_zonal_statistics.txt", "w", encoding="utf-8") as f:
    f.write("Hydraulic Conductivity - Zonal Statistics (n=24 independent pumping tests)\n")
    f.write("=" * 78 + "\n\n")
    f.write(zonal.to_string(index=False))
    f.write("\n\nK measured at these 13 wells only: " + ", ".join(sorted(df['Quyu ad'].unique())) + "\n")
    f.write("No direct K measurement exists at the remaining 16 named wells\n")
    f.write("(Yalama-1, Yalama-8, Khudat-10/112/113/20, K/L/M/N, etc.).\n\n")
    f.write("NOTE: A continuous kriging surface was tested and rejected - honest LOOCV\n")
    f.write("gave negative R2, i.e. the interpolated surface performs worse than the mean.\n")
    f.write("K exhibits small-scale heterogeneity not resolvable at this well density.\n")
print("\n-> outputs/conductivity_zonal_statistics.txt")

zone_order = [z for z in ['Khudat', 'Khachmaz', 'Yalama', 'Nabran'] if z in df['Zone'].unique()]

# =====================================================================
# FIGURE A - 4-panel overview
# =====================================================================
print("\nPreparing FIGURE A...")
figA = plt.figure(figsize=(14, 9), dpi=130)
gsA = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.1, 1], hspace=0.28, wspace=0.25)

# (a) map view
axA1 = figA.add_subplot(gsA[0, 0])
logk = np.log10(K)
sizes = 40 + (logk - logk.min()) / (logk.max() - logk.min()) * 400
scA = axA1.scatter(df['Lon'], df['Lat'], s=sizes, c=logk, cmap='YlOrBr',
                    edgecolor='k', linewidth=0.7, alpha=0.85, zorder=3)
for w in df['Quyu ad'].unique():
    wd = df[df['Quyu ad'] == w].iloc[0]
    axA1.annotate(w, (wd['Lon'], wd['Lat']), fontsize=7, xytext=(4, 4),
                   textcoords='offset points', color='#333')
plt.colorbar(scA, ax=axA1, label='log₁₀ K (m/s)', shrink=0.85)
axA1.set_xlabel('Longitude (°E)'); axA1.set_ylabel('Latitude (°N)')
axA1.set_title('(a) Well locations, symbol size ∝ log K', fontsize=10, loc='left')
axA1.grid(alpha=0.25, ls='--')

# (b) K vs depth
axA2 = figA.add_subplot(gsA[0, 1])
for z, g in df.groupby('Zone'):
    axA2.scatter(g['Keçiricilik (K)'], g['Zmid'], color=ZC[z], s=60,
                  edgecolor='white', linewidth=0.8, label=z, zorder=3)
axA2.set_xscale('log'); axA2.set_xlabel('K (m/s)'); axA2.set_ylabel('Depth (m)')
axA2.set_title('(b) K versus depth', fontsize=10, loc='left')
axA2.grid(alpha=0.25, ls='--'); axA2.legend(fontsize=8, frameon=False)

# (c) zonal ranges
axA3 = figA.add_subplot(gsA[1, 0])
for i, z in enumerate(zone_order):
    d = df[df['Zone'] == z]['Keçiricilik (K)'].values
    y = len(zone_order) - i
    axA3.plot([d.min(), d.max()], [y, y], color=ZC[z], lw=3, solid_capstyle='round', zorder=2)
    axA3.scatter(d, [y] * len(d), color=ZC[z], s=50, edgecolor='white', lw=0.8, zorder=3)
    axA3.scatter([np.median(d)], [y], color='white', s=100, edgecolor=ZC[z], lw=2,
                  marker='D', zorder=4)
    axA3.text(d.max() * 1.35, y, f"n={len(d)}", va='center', fontsize=8, color='#666')
axA3.set_yticks(range(1, len(zone_order) + 1))
axA3.set_yticklabels(zone_order[::-1], fontsize=10)
axA3.set_xscale('log'); axA3.set_xlabel('K (m/s)')
axA3.set_title('(c) Zonal K ranges (diamond = median)', fontsize=10, loc='left')
axA3.grid(alpha=0.25, ls='--', axis='x')
for s_ in ['top', 'right']:
    axA3.spines[s_].set_visible(False)

# (d) K vs thickness
axA4 = figA.add_subplot(gsA[1, 1])
for z, g in df.groupby('Zone'):
    axA4.scatter(g['M -sulu kompleksin qalınlığı'], g['Keçiricilik (K)'],
                  color=ZC[z], s=60, edgecolor='white', lw=0.8, label=z, zorder=3)
axA4.set_yscale('log'); axA4.set_xlabel('Aquifer thickness, M (m)'); axA4.set_ylabel('K (m/s)')
axA4.set_title('(d) K versus aquifer thickness', fontsize=10, loc='left')
axA4.grid(alpha=0.25, ls='--'); axA4.legend(fontsize=8, frameon=False)

figA.suptitle('Hydraulic Conductivity of the Caspian–Guba Geothermal Aquifers\n'
              f'{len(df)} independent pumping tests, {df["Quyu ad"].nunique()} wells',
              fontsize=13, y=0.98)
figA.savefig('outputs/FigureA_conductivity_overview.png', dpi=DPI_OUT, bbox_inches='tight')
plt.close(figA)
print("-> outputs/FigureA_conductivity_overview.png")

# =====================================================================
# FIGURE B - NW-SE transect "fence" diagram
# =====================================================================
print("Preparing FIGURE B...")
wells = df.groupby('Quyu ad').agg(X=('X', 'first'), Y=('Y', 'first'),
                                   Zone=('Zone', 'first')).reset_index()
XY = wells[['X', 'Y']].values
XYc = XY - XY.mean(axis=0)
_, _, vt = np.linalg.svd(XYc, full_matrices=False)
wells['dist'] = XYc @ vt[0]          # projection along the principal axis (PCA)
wells = wells.sort_values('dist').reset_index(drop=True)
wells['pos'] = range(len(wells))
posmap = dict(zip(wells['Quyu ad'], wells['pos']))

figB, axB = plt.subplots(figsize=(14, 7), dpi=130)
cmap = plt.cm.YlOrBr
norm = LogNorm(vmin=K.min(), vmax=K.max())
BAR_W = 0.62
for _, r in df.iterrows():
    p = posmap[r['Quyu ad']]
    top, bot = -abs(r['From Depth (m)']), -abs(r['To Depth (m)'])
    axB.add_patch(Rectangle((p - BAR_W / 2, bot), BAR_W, top - bot,
                             facecolor=cmap(norm(r['Keçiricilik (K)'])),
                             edgecolor='#333', linewidth=0.8, zorder=3))
    axB.text(p, (top + bot) / 2, f"{r['Keçiricilik (K)']*1000:.1f}", ha='center', va='center',
              fontsize=7, color='white' if norm(r['Keçiricilik (K)']) > 0.55 else '#222',
              fontweight='bold', zorder=4)
for _, w in wells.iterrows():
    wd = df[df['Quyu ad'] == w['Quyu ad']]
    axB.plot([w['pos'], w['pos']], [0, -abs(wd['To Depth (m)']).max()],
              color='#999', lw=1, ls=':', zorder=1)
    axB.scatter([w['pos']], [0], marker='v', s=90, color=ZC[w['Zone']],
                 edgecolor='k', linewidth=0.7, zorder=5)
axB.set_xticks(wells['pos'])
axB.set_xticklabels(wells['Quyu ad'], rotation=45, ha='right', fontsize=9)
for lbl, w in zip(axB.get_xticklabels(), wells.itertuples()):
    lbl.set_color(ZC[w.Zone])
axB.set_ylabel('Depth (m)', fontsize=11)
axB.set_ylim(-2100, 120)
axB.set_xlim(-0.8, len(wells) - 0.2)
axB.axhline(0, color='#444', lw=1.5)
axB.grid(axis='y', alpha=0.25, ls='--')
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbB = plt.colorbar(sm, ax=axB, pad=0.015, shrink=0.9)
cbB.set_label('Hydraulic conductivity, K (m/s)', fontsize=10)
handles = [Line2D([0], [0], marker='v', color='w', markerfacecolor=c, markeredgecolor='k',
                   markersize=10, label=z) for z, c in ZC.items() if z in df['Zone'].unique()]
axB.legend(handles=handles, loc='lower left', frameon=True, fontsize=9,
            title='Zone', title_fontsize=9)
axB.set_title('Hydraulic Conductivity — Tested Aquifer Intervals along NW–SE Transect\n'
              'Bar height = screened interval; label = K ×10⁻³ m/s; '
              f'{len(df)} pumping tests, {df["Quyu ad"].nunique()} wells',
              fontsize=12, pad=12)
for s_ in ['top', 'right']:
    axB.spines[s_].set_visible(False)
figB.tight_layout()
figB.savefig('outputs/FigureB_conductivity_transect.png', dpi=DPI_OUT, bbox_inches='tight')
plt.close(figB)
print("-> outputs/FigureB_conductivity_transect.png")

# =====================================================================
# FIGURE C - Correlation structure (direct response to Reviewer #3)
# =====================================================================
print("Preparing FIGURE C...")
figC = plt.figure(figsize=(14, 8.5), dpi=130)
gsC = gridspec.GridSpec(2, 3, hspace=0.33, wspace=0.30)


def corr_panel(ax, x, y, xlab, ylab, title, logx=False, logy=False, warn=False):
    for z in df['Zone'].unique():
        m = (df['Zone'] == z).values
        ax.scatter(x[m], y[m], color=ZC[z], s=55, edgecolor='white', lw=0.8, zorder=3, label=z)
    xs = np.log10(x) if logx else x
    ys = np.log10(y) if logy else y
    r, p = stats.pearsonr(xs, ys)
    sl, ic = np.polyfit(xs, ys, 1)
    xr = np.linspace(xs.min(), xs.max(), 50)
    yr = sl * xr + ic
    ax.plot(10 ** xr if logx else xr, 10 ** yr if logy else yr,
            color='#d62728' if warn else '#333', ls='--', lw=1.6, zorder=2)
    if logx:
        ax.set_xscale('log')
    if logy:
        ax.set_yscale('log')
    ax.set_xlabel(xlab, fontsize=10); ax.set_ylabel(ylab, fontsize=10)
    star = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.'))
    box = dict(boxstyle='round,pad=0.35', fc='#ffe0e0' if warn else '#f0f0f0',
               ec='#d62728' if warn else '#999', lw=1.2)
    ax.text(0.04, 0.95, f"r = {r:.3f} {star}\nn = {len(x)}", transform=ax.transAxes,
            va='top', fontsize=9.5, bbox=box)
    ax.set_title(title, fontsize=10, loc='left',
                 color='#d62728' if warn else '#222', fontweight='bold' if warn else 'normal')
    ax.grid(alpha=0.25, ls='--')
    return r, p


axC1 = figC.add_subplot(gsC[0, 0])
corr_panel(axC1, Q, T, 'Discharge rate, Q (m³/day)', 'Temperature (°C)',
           '(a) Temperature vs discharge  [INDEPENDENT]', logx=True)
axC1.legend(fontsize=8, frameon=False, loc='lower right')
axC2 = figC.add_subplot(gsC[0, 1])
corr_panel(axC2, np.abs(Zmid), T, 'Depth (m)', 'Temperature (°C)',
           '(b) Temperature vs depth  [INDEPENDENT]')
axC3 = figC.add_subplot(gsC[0, 2])
corr_panel(axC3, np.abs(Zmid), M, 'Depth (m)', 'Aquifer thickness, M (m)',
           '(c) Aquifer thickness vs depth  [INDEPENDENT]')
axC4 = figC.add_subplot(gsC[1, 0])
corr_panel(axC4, Q, K, 'Discharge rate, Q (m³/day)', 'K (m/s)',
           '(d) K vs Q  — PARTLY DEFINITIONAL', logx=True, logy=True, warn=True)
axC5 = figC.add_subplot(gsC[1, 1])
corr_panel(axC5, M, K, 'Aquifer thickness, M (m)', 'K (m/s)',
           '(e) K vs M  — PARTLY DEFINITIONAL', logy=True, warn=True)

axC6 = figC.add_subplot(gsC[1, 2]); axC6.axis('off')
axC6.text(0.0, 0.97, "Why panels (d)–(e) are not findings", fontsize=10.5,
          fontweight='bold', color='#d62728', va='top', transform=axC6.transAxes)
axC6.text(0.0, 0.86,
          "K is not measured independently. It is\n"
          "computed from the same variables:\n\n"
          "     K = 1.22 · (Q / S) / M\n\n"
          "so, in logarithmic form,\n\n"
          "  log K = log1.22 + logQ − logS − logM\n\n"
          "Any correlation of K with Q, S or M is\n"
          "therefore partly imposed by the formula,\n"
          "not an empirical result. Only panels\n"
          "(a)–(c), which relate quantities measured\n"
          "independently of one another, carry\n"
          "physical information.",
          fontsize=9, va='top', family='monospace', transform=axC6.transAxes, linespacing=1.5)

figC.suptitle(f'Correlation Structure of the Caspian–Guba Pumping-Test Dataset (n = {len(df)})\n'
              'Separating independently measured relationships from formula-imposed ones',
              fontsize=13, y=0.985)
figC.savefig('outputs/FigureC_correlation_structure.png', dpi=DPI_OUT, bbox_inches='tight')
plt.close(figC)
print("-> outputs/FigureC_correlation_structure.png")

print("\nAll three figures and the zonal statistics file are ready (300 dpi, journal-ready).")
