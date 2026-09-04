import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.patches import Rectangle, Patch
from scipy.interpolate import griddata
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import warnings
warnings.filterwarnings('ignore')

study_area_lon_min = 48.57
study_area_lon_max = 48.84
study_area_lat_min = 41.47
study_area_lat_max = 41.82

print("Loading energy data...")
os.makedirs('outputs', exist_ok=True)
energy_df = pd.read_csv('outputs/energy_map_data.csv', sep=',', engine='python')
energy_df.columns = energy_df.columns.str.strip()

rename_map = {}
for col in energy_df.columns:
    if 'Well' in col:
        rename_map[col] = 'Well'
    elif 'Lat' in col:
        rename_map[col] = 'Lat'
    elif 'Lon' in col:
        rename_map[col] = 'Lon'
    elif col == 'Energy_kWh':
        rename_map[col] = 'Value'
energy_df.rename(columns=rename_map, inplace=True)

for col in ['Lat', 'Lon', 'Value']:
    energy_df[col] = pd.to_numeric(energy_df[col], errors='coerce')
energy_df.dropna(inplace=True)

print(f"{len(energy_df)} data points loaded")
print(f"Columns: {energy_df.columns.tolist()}")
print(energy_df.head())

print("\nComputing interpolation...")
def create_grid(df, resolution=100):
    xi = np.linspace(df['Lon'].min(), df['Lon'].max(), resolution)
    yi = np.linspace(df['Lat'].min(), df['Lat'].max(), resolution)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(points=df[['Lon', 'Lat']].values, values=df['Value'].values,
                 xi=(X, Y), method='linear')
    return X, Y, Z

X_energy, Y_energy, Z_energy = create_grid(energy_df)
print("Interpolation completed successfully!")
print(f"Energy range: {energy_df['Value'].min():.1f} - {energy_df['Value'].max():.1f}")

fig = plt.figure(figsize=(16, 10), dpi=100)
ax_main = fig.add_axes([0.05, 0.05, 0.65, 0.9])
ax_inset = fig.add_axes([0.72, 0.55, 0.25, 0.35], projection=ccrs.PlateCarree())

contour = ax_main.contourf(X_energy, Y_energy, Z_energy, levels=15, cmap='plasma_r', alpha=0.8)
contour_lines = ax_main.contour(X_energy, Y_energy, Z_energy, levels=15, colors='black',
                                 linewidths=0.5, alpha=0.6)
clabels = ax_main.clabel(contour_lines, inline=True, fontsize=9, fmt='%.0f', colors='black')
for label in clabels:
    label.set_fontweight('bold')

ax_main.scatter(energy_df['Lon'], energy_df['Lat'], c='black', s=30, edgecolors='white',
                 linewidth=0.5, label='Measurement points', zorder=5)
for i, row in energy_df.iterrows():
    ax_main.text(row['Lon'], row['Lat'], row['Well'],
                 fontsize=10, fontweight='bold', ha='center', va='bottom',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

cbar = plt.colorbar(contour, ax=ax_main, shrink=0.8, pad=0.02)
cbar.set_label('Electrical Power (kW)', fontsize=12, rotation=270, labelpad=20)

ax_main.set_xlabel('Longitude', fontsize=11)
ax_main.set_ylabel('Latitude', fontsize=11)
ax_main.set_title('Caspian - Guba region (Electrical Power)',
                   fontsize=14, fontweight='bold', pad=20)
ax_main.grid(True, linestyle='--', alpha=0.3)
ax_main.legend(loc='lower right', fontsize=9)
ax_main.set_aspect('equal')

ax_inset.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.6)
ax_inset.add_feature(cfeature.OCEAN, facecolor='lightblue', alpha=0.6)
ax_inset.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5, alpha=0.7)
ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.8, alpha=0.8)
ax_inset.add_feature(cfeature.LAKES, facecolor='lightblue', alpha=0.6)
ax_inset.set_extent([47.8, 49.5, 40.8, 42.0], crs=ccrs.PlateCarree())
ax_inset.set_title('Regional map', fontsize=10, fontweight='bold')
gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.3, color='gray',
                               alpha=0.5, linestyle=':')
gl_inset.top_labels = False
gl_inset.right_labels = False

rect = Rectangle((study_area_lon_min, study_area_lat_min),
                  study_area_lon_max - study_area_lon_min,
                  study_area_lat_max - study_area_lat_min,
                  linewidth=2, edgecolor='red', facecolor='none',
                  alpha=0.9, label='Research area',
                  transform=ccrs.PlateCarree(), zorder=5)
ax_inset.add_patch(rect)
legend_elements = [Patch(facecolor='none', edgecolor='red', linewidth=2, alpha=0.9, label='Research area')]
ax_inset.legend(handles=legend_elements, loc='upper left', fontsize=7)
ax_inset.tick_params(labelsize=7)

plt.savefig('outputs/Figure6_energy_map_corrected.png', dpi=200, bbox_inches='tight')
print("\nFigure saved (in the outputs/ directory): outputs/Figure6_energy_map_corrected.png")
plt.show()
print("\nFigure saved")

print(f"\nEnergy Statistics:")
print(f"   Min: {energy_df['Value'].min():.1f} kW")
print(f"   Max: {energy_df['Value'].max():.1f} kW")
print(f"   Mean: {energy_df['Value'].mean():.1f} kW")
print(f"   Number of points: {len(energy_df)}")
