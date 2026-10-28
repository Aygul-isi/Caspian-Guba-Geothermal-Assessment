"""
=============================================================================
Study Area Map - Caspian-Guba region, Azerbaijan
=============================================================================
Draws the study-area location map: the Caspian-Guba boundary polygon on a
main map (with a geodesic scale bar) and a regional inset map (Caucasus /
Caspian Sea) in which the study area is highlighted in red.

The boundary shapefile may be in any CRS (the original is UTM Zone 39N);
it is reprojected to WGS84 (EPSG:4326) for plotting.

Input:  data/caspian_guba_boundary.shp (+ .dbf/.shx/.prj)
Output: outputs/study_area_map.png (300 dpi)
=============================================================================
"""

import os

# The non-interactive 'Agg' backend must be selected before pyplot is imported
import matplotlib
matplotlib.use('Agg')

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import numpy as np
from pyproj import Geod

# --- USER SETTINGS ---
SHAPEFILE_PATH = 'data/caspian_guba_boundary.shp'
OUTPUT_PATH = 'outputs/study_area_map.png'
DPI_OUT = 300


def add_scale_bar(ax, length_km=None, location=(0.05, 0.04), bar_height_frac=0.012):
    """
    Draw a geodesically accurate linear scale bar on a cartopy GeoAxes
    (PlateCarree), in the classic alternating black/white cartographic style.

    ax             : cartopy GeoAxes (PlateCarree) to draw the scale bar on
    length_km      : total bar length in km. If None, a "round" value
                     (1/2/5/10 x 10^n) close to ~1/4 of the map width is chosen.
    location       : position of the bar's lower-left corner, as a fraction
                     (x_frac, y_frac) of the current map extent, in 0-1.
    bar_height_frac: bar thickness, as a fraction of the map's vertical extent.
    """
    geod = Geod(ellps='WGS84')
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())
    lat_mid = (lat_min + lat_max) / 2.0

    # True (geodesic) east-west width of the map at the mid latitude, in km
    _, _, map_width_m = geod.inv(lon_min, lat_mid, lon_max, lat_mid)
    map_width_km = map_width_m / 1000.0

    if length_km is None:
        target = map_width_km * 0.25
        magnitude = 10 ** np.floor(np.log10(target))
        length_km = magnitude
        for candidate in (1, 2, 5, 10):
            length_km = candidate * magnitude
            if length_km >= target:
                break

    n_segments = 4
    seg_km = length_km / n_segments

    x0 = lon_min + location[0] * (lon_max - lon_min)
    y0 = lat_min + location[1] * (lat_max - lat_min)
    bar_height = (lat_max - lat_min) * bar_height_frac

    # Alternating black/white segments, left to right, by geodesic distance
    lon_cursor = x0
    for i in range(n_segments):
        lon_next, _, _ = geod.fwd(lon_cursor, y0, 90.0, seg_km * 1000.0)
        seg_color = 'black' if i % 2 == 0 else 'white'
        ax.add_patch(Rectangle(
            (lon_cursor, y0), lon_next - lon_cursor, bar_height,
            facecolor=seg_color, edgecolor='black', linewidth=0.8,
            transform=ccrs.PlateCarree(), zorder=10
        ))
        lon_cursor = lon_next

    # End labels: "0" and the total length (km)
    ax.text(x0, y0 + bar_height * 1.6, '0', ha='center', va='bottom',
            fontsize=8, transform=ccrs.PlateCarree(), zorder=11)
    ax.text(lon_cursor, y0 + bar_height * 1.6, f'{length_km:.0f} km',
            ha='center', va='bottom', fontsize=8, transform=ccrs.PlateCarree(),
            zorder=11)


# 1. Read the study-area boundary and reproject it to WGS84 (lon/lat)
gdf = gpd.read_file(SHAPEFILE_PATH)
gdf_wgs84 = gdf.to_crs(epsg=4326)

# 2. Main map of the Caspian-Guba region
fig = plt.figure(figsize=(12, 10))
ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection=ccrs.PlateCarree())

gdf_wgs84.plot(ax=ax_main, facecolor='orange', edgecolor='black', hatch='////', linewidth=1)

# Fit the map extent to the boundary, with a small margin
bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
margin = 0.05
ax_main.set_extent([bounds[0] - margin, bounds[2] + margin,
                    bounds[1] - margin, bounds[3] + margin], crs=ccrs.PlateCarree())

gl = ax_main.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Scale bar in the lower-left corner
add_scale_bar(ax_main, location=(0.05, 0.06))

ax_main.set_title("", fontsize=16, fontweight='bold', pad=20)

# 3. Regional inset map (Caucasus and the Caspian Sea)
ax_inset = fig.add_axes([0.65, 0.45, 0.3, 0.3], projection=ccrs.PlateCarree())

ax_inset.add_feature(cfeature.LAND, facecolor='lightgray')
ax_inset.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_inset.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5)
ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Approximate extent: [Min_Lon, Max_Lon, Min_Lat, Max_Lat]
ax_inset.set_extent([38, 56, 36, 45], crs=ccrs.PlateCarree())
ax_inset.set_title("Regional map", fontsize=12, fontweight='bold')

gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle=':')
gl_inset.top_labels = False
gl_inset.right_labels = False
gl_inset.xlabel_style = {'size': 8}
gl_inset.ylabel_style = {'size': 8}

# 4. Highlight the study area in red on the inset map, with a bounding box
gdf_wgs84.plot(ax=ax_inset, facecolor='red', edgecolor='darkred', linewidth=1, zorder=5)

rect = Rectangle((bounds[0], bounds[1]), bounds[2]-bounds[0], bounds[3]-bounds[1],
                 linewidth=1.5, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree(), zorder=6)
ax_inset.add_patch(rect)

# Save the figure
os.makedirs('outputs', exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=DPI_OUT, bbox_inches='tight')
print(f"-> {OUTPUT_PATH}")
