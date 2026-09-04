"""
=============================================================================
3D Well Discharge Rate Model - Local Ordinary Kriging
=============================================================================
Same validated recipe used throughout this repository: duplicate removal +
log10 transform + LOOCV model selection + local (moving-window) kriging +
uncertainty quantification.

NOTE: Discharge rate is not a "clean" geological field in the way that K or
temperature are - it also depends on the specific pumping-test conditions
of each well (test duration, targeted drawdown), not only on aquifer
properties. It is therefore more accurate to present this map as "discharge
rate observed under the prevailing test conditions" rather than as a strict
prediction of aquifer productivity. This caveat is stated explicitly in the
manuscript (Section 4.3 / Section 5.2).

Data source: discharge_rate_intervals.xlsx (Lon, Lat, from_depth_m,
to_depth_m, Discharge_rate)
  - 165 valid rows / 29 wells
  - The mid-depth (Z) of each interval is computed automatically.

Required libraries: pip install gstools scikit-learn pyproj geopandas plotly
=============================================================================
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
import gstools as gs
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
import geopandas as gpd
import webbrowser
import os

# =====================================================================
DISCHARGE_FILE = 'data/discharge_rate_intervals.xlsx'
BOUNDARY_SHAPEFILE = 'data/caspian_guba_boundary.shp'
WELL_NAMES_FILE = 'data/well_names.xlsx'
NX, NY, NZ = 40, 40, 40
K_NEIGHBORS = 15          # reduced from 30: with only 29 wells, a larger
                          # neighbourhood would span most of the dataset
CANDIDATE_MODELS = {
    'Spherical': gs.Spherical,
    'Exponential': gs.Exponential,
    # Gaussian model excluded: its infinitely smooth curve makes the local
    # kriging system ill-conditioned; in repeated tests across all three
    # fields (temperature, hydraulic conductivity, discharge rate) it
    # always performed worse than Spherical/Exponential and occasionally
    # caused numerical overflow.
}
LOG_TRANSFORM = True      # skewness test: raw=1.68 -> log=-0.53, log is recommended
# =====================================================================

transformer = Transformer.from_crs("epsg:4326", "epsg:32639", always_xy=True)

print("Loading discharge-rate data...")
df = pd.read_excel(DISCHARGE_FILE)
df = df.dropna(subset=['Lon', 'Lat', 'from_depth_m', 'to_depth_m', 'Discharge_rate'])
print(f"Loaded {len(df)} valid rows (rows with missing values removed).")

df['X_UTM'], df['Y_UTM'] = transformer.transform(df['Lon'].values, df['Lat'].values)
df['Z_mid'] = (df['from_depth_m'] + df['to_depth_m']) / 2

x = df['X_UTM'].values
y = df['Y_UTM'].values
z = df['Z_mid'].values
discharge = df['Discharge_rate'].values
coords_raw = np.column_stack([x, y, z])

if LOG_TRANSFORM:
    work_vals = np.log10(discharge)
    print("-> Kriging is performed on log10(discharge rate).")
else:
    work_vals = discharge.copy()

coords, inv = np.unique(coords_raw, axis=0, return_inverse=True)
n_dupes = len(coords_raw) - len(coords)
if n_dupes > 0:
    work_vals = np.array([work_vals[inv == i].mean() for i in range(len(coords))])
    discharge = np.array([discharge[inv == i].mean() for i in range(len(coords))])
    print(f"NOTE: {n_dupes} duplicate coordinates were reduced to their mean value "
          f"({len(coords_raw)} -> {len(coords)}).")
else:
    print("No duplicate coordinates found.")

x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
tree = cKDTree(coords)


def local_ok_predict_batch(query_pts, coords, vals, tree, model, k=15, exclude_self=False):
    n_query = len(query_pts)
    preds = np.empty(n_query)
    variances = np.empty(n_query)
    k = min(k, len(coords) - (1 if exclude_self else 0))
    k_search = k + 1 if exclude_self else k
    dists, idxs = tree.query(query_pts, k=k_search)
    for qi in range(n_query):
        idx = np.atleast_1d(idxs[qi])
        if exclude_self:
            self_mask = np.all(coords[idx] == query_pts[qi], axis=1)
            idx = idx[~self_mask][:k]
        lc = coords[idx]
        lv = vals[idx]
        nl = len(idx)
        D = np.linalg.norm(lc[:, None, :] - lc[None, :, :], axis=-1)
        G = model.variogram(D)
        Kmat = np.ones((nl + 1, nl + 1))
        Kmat[:nl, :nl] = G
        Kmat[-1, -1] = 0
        d = np.linalg.norm(lc - query_pts[qi], axis=1)
        kv = np.ones(nl + 1)
        kv[:nl] = model.variogram(d)
        weights, *_ = np.linalg.lstsq(Kmat, kv, rcond=None)
        preds[qi] = np.dot(weights[:nl], lv)
        variances[qi] = np.dot(weights[:nl], kv[:nl]) + weights[-1]
    return preds, np.clip(variances, 0, None)


print(f"\nFitting variogram + LOOCV on {len(coords)} unique points ({K_NEIGHBORS} neighbours)...")
bin_center, gamma_emp = gs.vario_estimate((x, y, z), work_vals, latlon=False)
var_est = np.var(work_vals)
nugget_floor = 0.01 * var_est

results = {}
for name, ModelClass in CANDIDATE_MODELS.items():
    model = ModelClass(dim=3)
    model.fit_variogram(bin_center, gamma_emp, nugget=True)
    if model.nugget < nugget_floor:
        model.nugget = nugget_floor

    preds_work, _ = local_ok_predict_batch(coords, coords, work_vals, tree, model,
                                            k=K_NEIGHBORS, exclude_self=True)
    rmse_log = np.sqrt(mean_squared_error(work_vals, preds_work))
    r2_log = r2_score(work_vals, preds_work)

    # Safety net: ill-conditioned local systems can rarely produce
    # unreasonably large log-scale values (lstsq prevents singularity, but
    # the result can still be unphysical). To stop 10^x from overflowing,
    # predictions are clipped to a finite range well beyond the real data
    # range.
    clip_lo, clip_hi = work_vals.min() - 5, work_vals.max() + 5
    n_clipped = np.sum((preds_work < clip_lo) | (preds_work > clip_hi))
    if n_clipped > 0:
        print(f"    [WARNING] {n_clipped} predictions fell outside the physical range and were clipped.")
    preds_work = np.clip(preds_work, clip_lo, clip_hi)

    preds_real = 10 ** preds_work if LOG_TRANSFORM else preds_work
    true_real = discharge
    rmse_real = np.sqrt(mean_squared_error(true_real, preds_real))
    mae_real = mean_absolute_error(true_real, preds_real)
    r2_real = r2_score(true_real, preds_real)

    results[name] = {'model': model, 'rmse_log': rmse_log, 'r2_log': r2_log,
                      'rmse_real': rmse_real, 'mae_real': mae_real, 'r2_real': r2_real}
    print(f"  {name:12s}  [log10]  RMSE={rmse_log:.3f} R²={r2_log:.3f}   "
          f"[real m³/day]  RMSE={rmse_real:.1f}  MAE={mae_real:.1f}  R²={r2_real:.3f}")

best_name = min(results, key=lambda m: results[m]['rmse_log'])
best = results[best_name]
print(f"\n>>> Selected model: '{best_name}' (log10 RMSE={best['rmse_log']:.3f}, R²={best['r2_log']:.3f})")

with open("validation_report_discharge_rate.txt", "w", encoding="utf-8") as f:
    f.write("3D Local Ordinary Kriging Validation Report - Discharge Rate Model\n")
    f.write("=" * 76 + "\n\n")
    f.write(f"Number of observations: {len(discharge)}\n")
    f.write(f"Duplicate points averaged: {n_dupes}\n")
    f.write(f"Log10 transform applied: {LOG_TRANSFORM}\n")
    f.write(f"Neighbors (K): {K_NEIGHBORS}\n\n")
    for name, res in results.items():
        f.write(f"Variogram model: {name}\n")
        f.write(f"  [log10 scale]      RMSE={res['rmse_log']:.4f}  R2={res['r2_log']:.4f}\n")
        f.write(f"  [real, m3/day]     RMSE={res['rmse_real']:.2f}  MAE={res['mae_real']:.2f}  R2={res['r2_real']:.4f}\n\n")
    f.write(f"Selected model: {best_name}\n")
    f.write("\nNOTE: Discharge rate also depends on well-test conditions (choice of\n"
            "drawdown, test duration), not only on aquifer properties - this map should\n"
            "be presented as 'discharge rate observed under test conditions', not as a\n"
            "strict prediction of aquifer productivity.\n")
print("Validation report written -> validation_report_discharge_rate.txt")

# Grid prediction; also exports a function for well-specific prediction in
# downstream scripts
xi = np.linspace(x.min(), x.max(), NX)
yi = np.linspace(y.min(), y.max(), NY)
zi = np.linspace(z.min(), z.max(), NZ)
XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
grid_pts = np.column_stack([XI.ravel(), YI.ravel(), ZI.ravel()])

print(f"\nRunning kriging over the grid ({len(grid_pts)} points)...")
grid_pred_work, grid_var_work = local_ok_predict_batch(grid_pts, coords, work_vals, tree,
                                                         best['model'], k=K_NEIGHBORS, exclude_self=False)
clip_lo, clip_hi = work_vals.min() - 5, work_vals.max() + 5
grid_pred_work = np.clip(grid_pred_work, clip_lo, clip_hi)
grid_discharge = (10 ** grid_pred_work if LOG_TRANSFORM else grid_pred_work).reshape(XI.shape)
grid_std_log = np.sqrt(grid_var_work).reshape(XI.shape)
print("Grid kriging completed.")

# Save results for the downstream script
os.makedirs("outputs", exist_ok=True)
np.savez("outputs/discharge_rate_kriging_model.npz",
         coords=coords, work_vals=work_vals, discharge=discharge,
         xi=xi, yi=yi, zi=zi, grid_discharge=grid_discharge, grid_std_log=grid_std_log,
         log_transform=LOG_TRANSFORM, best_model_name=best_name)
print("Model results saved -> outputs/discharge_rate_kriging_model.npz")

# ============================================================
# 4. Plotly 3D visualisation (same style as the other model scripts)
# ============================================================
fig = go.Figure()

fig.add_trace(go.Volume(
    x=XI.flatten(), y=YI.flatten(), z=ZI.flatten(),
    value=grid_discharge.flatten(),
    isomin=np.percentile(discharge, 5),
    isomax=np.percentile(discharge, 95),
    opacity=0.15, surface_count=20,
    colorscale='YlGnBu',
    colorbar=dict(title="Discharge rate (m³/day)", x=1.02, xanchor="left", len=0.7, yanchor="middle", orientation="v"),
    caps=dict(x_show=False, y_show=False, z_show=False),
    hovertemplate="X (UTM): %{x:.1f}<br>Y (UTM): %{y:.1f}<br>Depth: %{z:.1f} m<br>Discharge: %{value:.0f} m³/day<extra></extra>",
    name="Discharge Rate Estimate",
    visible=True
))

fig.add_trace(go.Volume(
    x=XI.flatten(), y=YI.flatten(), z=ZI.flatten(),
    value=grid_std_log.flatten(),
    isomin=np.percentile(grid_std_log, 5),
    isomax=np.percentile(grid_std_log, 95),
    opacity=0.15, surface_count=20,
    colorscale='Viridis',
    colorbar=dict(title="Kriging Std. Dev. (log10 units)", x=1.55, xanchor="left", len=0.7, yanchor="middle", orientation="v"),
    caps=dict(x_show=False, y_show=False, z_show=False),
    hovertemplate="X (UTM): %{x:.1f}<br>Y (UTM): %{y:.1f}<br>Depth: %{z:.1f} m<br>Uncertainty (σ, log10): %{value:.2f}<extra></extra>",
    name="Kriging Uncertainty",
    visible=False
))

print("Loading regional boundary shapefile...")
gdf = gpd.read_file(BOUNDARY_SHAPEFILE)
if gdf.crs.to_epsg() != 32639:
    gdf = gdf.to_crs(epsg=32639)

for geom in gdf.geometry:
    if geom.geom_type == 'LineString':
        xs, ys = map(np.array, geom.xy)
        zs = np.full_like(xs, 0)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                    line=dict(color='black', width=4), name="Regional Boundary Line"))
    elif geom.geom_type == 'Polygon':
        xs, ys = map(np.array, geom.exterior.xy)
        zs = np.full_like(xs, 0)
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode='lines',
                                    line=dict(color='black', width=3), name="Regional Polygon Area"))

print("Adding well locations...")
wells_df = pd.read_excel(WELL_NAMES_FILE)
wells_df['X'], wells_df['Y'] = transformer.transform(wells_df['Lon'].values, wells_df['Lat'].values)
wells_df['Z'] = 0

fig.add_trace(go.Scatter3d(
    x=wells_df['X'], y=wells_df['Y'], z=wells_df['Z'],
    mode='markers+text',
    marker=dict(size=4, color='black'),
    text=wells_df['Well_Name'],
    textposition="top center",
    textfont=dict(color='black', size=14),
    name="Well Locations"
))

n_traces = len(fig.data)
show_est = [True, False] + [True] * (n_traces - 2)
show_unc = [False, True] + [True] * (n_traces - 2)

fig.update_layout(
    title=(f"3D Well Discharge Rate Model (Local Ordinary Kriging, {best_name} variogram, K={K_NEIGHBORS})<br>"
           f"<sub>LOOCV (n={len(discharge)}, log10 scale): RMSE={best['rmse_log']:.3f}, R²={best['r2_log']:.3f}"
           f" | Discharge rate reflects well-test conditions, not purely aquifer properties</sub>"),
    scene=dict(xaxis_title='X (UTM, m)', yaxis_title='Y (UTM, m)', zaxis_title='Depth (m)',
               aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5)),
    width=1000, height=800, margin=dict(l=50, r=50, b=50, t=140),
    updatemenus=[dict(
        type="buttons", direction="down", showactive=True,
        x=1.3, y=1.1, xanchor="right", yanchor="top",
        buttons=[
            dict(label="Show Discharge Estimate", method="restyle", args=[{"visible": show_est}]),
            dict(label="Show Uncertainty (σ)", method="restyle", args=[{"visible": show_unc}]),
        ]
    )],
    legend=dict(tracegroupgap=10, itemclick="toggleothers", itemdoubleclick="toggle")
)

os.makedirs("outputs", exist_ok=True)
html_file = "outputs/3d_discharge_rate_model_kriging.html"
fig.write_html(html_file, auto_open=False)
print(f"\nModel successfully created: {html_file}")
if os.environ.get("OPEN_IN_BROWSER", "1") == "1":
    webbrowser.open('file://' + os.path.realpath(html_file))
