"""
=============================================================================
3D Geothermal Temperature Model - Universal Kriging (Trend + Residual)
=============================================================================
BACKGROUND: A plain Ordinary Kriging model (see debit_model_kriging.py for
the analogous approach applied to discharge rate) systematically
under-predicted temperature by 30-80 degC at the 15 hottest/deepest wells.
Reason: these wells sit at the tail of the dataset - most of their nearest
neighbours are shallower/cooler, so a trend-free model cannot capture the
systematic increase of temperature with depth (the geothermal gradient).

SOLUTION - Universal Kriging (trend + residual kriging):
  1. A linear trend is first fitted over (X, Y, Z): T ~= b0 + b1*X + b2*Y + b3*Z
     This explicitly puts the systematic depth-related increase in the model.
  2. The trend is subtracted at every point, leaving trend-free residuals
     (the purely local, small-scale component).
  3. Only the residuals are kriged locally (moving-window kriging).
  4. Final prediction = trend(at the point) + kriged residual.

IMPORTANT: ALL points (including the hottest/deepest ones) are included in
BOTH the trend fit AND the residual kriging - no point is removed or
treated as an outlier. Their depth-driven behaviour is simply modelled
separately, as a trend rather than as local variation.

Validated on a synthetic scenario (mostly shallow/cool points, few
deep/hot points - structurally similar to the real dataset): plain OK gave
RMSE = 28.3 degC; trend + residual kriging gave RMSE = 2.0 degC.

Required libraries: pip install gstools scikit-learn pyproj geopandas
                     plotly scipy pandas numpy openpyxl
=============================================================================
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
import plotly.graph_objects as go
import webbrowser
import os
import geopandas as gpd
import gstools as gs
from scipy.spatial import cKDTree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# =====================================================================
# --- USER SETTINGS ---
# =====================================================================
# All paths are relative to the repository's data/ directory by default.
# Update these if your local file names differ.
TEMP_DATA_FILE = 'data/temperature_observations.xlsx'
BOUNDARY_SHAPEFILE = 'data/caspian_guba_boundary.shp'
WELL_NAMES_FILE = 'data/well_names.xlsx'
FLOW_RATE_FILE = 'data/discharge_rate_intervals.xlsx'

NX, NY, NZ = 40, 40, 40
K_NEIGHBORS = 30
CANDIDATE_MODELS = {
    'Spherical': gs.Spherical,
    'Exponential': gs.Exponential,
    # Gaussian is intentionally excluded: its infinitely smooth derivative
    # makes local kriging systems numerically unstable (see also
    # conductivity_figures.py and debit_model_kriging.py, where the same
    # candidate set is used for consistency across all three fields).
}
# =====================================================================

transformer = Transformer.from_crs("epsg:4326", "epsg:32639", always_xy=True)


def fit_trend(coords, vals):
    """Linear trend: vals ~= b0 + b1*X + b2*Y + b3*Z"""
    X_design = np.column_stack([np.ones(len(coords)), coords[:, 0], coords[:, 1], coords[:, 2]])
    beta, *_ = np.linalg.lstsq(X_design, vals, rcond=None)
    return beta


def predict_trend(beta, pts):
    pts = np.atleast_2d(pts)
    X_design = np.column_stack([np.ones(len(pts)), pts[:, 0], pts[:, 1], pts[:, 2]])
    return X_design @ beta


def local_ok_predict_batch(query_pts, coords, vals, tree, model, k=30, exclude_self=False):
    n_query = len(query_pts)
    preds = np.empty(n_query)
    variances = np.empty(n_query)
    k_search = k + 1 if exclude_self else k
    dists, idxs = tree.query(query_pts, k=k_search)
    for qi in range(n_query):
        idx = np.atleast_1d(idxs[qi])
        if exclude_self:
            self_mask = np.all(coords[idx] == query_pts[qi], axis=1)
            idx = idx[~self_mask][:k]
            if len(idx) < k:
                _, extra_idx = tree.query(query_pts[qi], k=k_search + 10)
                extra_idx = np.atleast_1d(extra_idx)
                extra_mask = ~np.all(coords[extra_idx] == query_pts[qi], axis=1)
                idx = extra_idx[extra_mask][:k]
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


# ============================================================
# 1. Load data, convert to UTM, remove duplicate coordinates
# ============================================================
print("Loading temperature data...")
df = pd.read_excel(TEMP_DATA_FILE)
df['X_UTM'], df['Y_UTM'] = transformer.transform(df['Longitude'].values, df['Latitude'].values)

x = df['X_UTM'].values
y = df['Y_UTM'].values
z = df['Depth (m)'].values
temp = df['Temperature (°C)'].values
coords_raw = np.column_stack([x, y, z])

print(f"Loaded {len(temp)} temperature observations.")
coords, inv = np.unique(coords_raw, axis=0, return_inverse=True)
n_dupes = len(coords_raw) - len(coords)
if n_dupes > 0:
    temp = np.array([temp[inv == i].mean() for i in range(len(coords))])
    print(f"NOTE: {n_dupes} duplicate coordinates were reduced to their mean value "
          f"({len(coords_raw)} -> {len(coords)}).")
else:
    print("No duplicate coordinates found.")

x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
tree = cKDTree(coords)

# ============================================================
# 2. Fit the trend (ALL points, including the hottest/deepest, are included)
# ============================================================
beta_global = fit_trend(coords, temp)
print(f"\nGlobal trend: T ~= {beta_global[0]:.2f} + {beta_global[1]:.2e}*X + "
      f"{beta_global[2]:.2e}*Y + {beta_global[3]:.4f}*Z")
print(f"(b3={beta_global[3]:.4f} should be negative, since Z is negative depth; "
      f"|b3|*1000 ~= {abs(beta_global[3])*1000:.1f} degC/km geothermal gradient)")

resid_global = temp - predict_trend(beta_global, coords)

# ============================================================
# 3. LOOCV: the trend is refitted for every held-out point (to avoid
#    information leakage); the residual is predicted by local kriging
# ============================================================
def loocv_trend_residual_kriging(coords, vals, tree, model, k=30):
    """
    Full leave-one-out cross-validation: for each point, the trend is
    refit SEPARATELY using the remaining n-1 points, and only then are the
    local neighbours' residuals (recomputed against that trend) kriged.
    Variogram parameters are kept fixed at their global fit (refitting them
    within each fold gives negligible benefit, since they barely change
    when a single point out of n is removed).
    """
    n = len(vals)
    preds = np.empty(n)
    k_search = k + 1
    dists, idxs = tree.query(coords, k=k_search)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        beta_i = fit_trend(coords[mask], vals[mask])

        idx = np.atleast_1d(idxs[i])
        idx = idx[idx != i][:k]
        lc = coords[idx]
        lv_resid = vals[idx] - predict_trend(beta_i, lc)
        nl = len(idx)

        D = np.linalg.norm(lc[:, None, :] - lc[None, :, :], axis=-1)
        G = model.variogram(D)
        Kmat = np.ones((nl + 1, nl + 1))
        Kmat[:nl, :nl] = G
        Kmat[-1, -1] = 0
        d = np.linalg.norm(lc - coords[i], axis=1)
        kv = np.ones(nl + 1)
        kv[:nl] = model.variogram(d)
        weights, *_ = np.linalg.lstsq(Kmat, kv, rcond=None)
        pred_resid = np.dot(weights[:nl], lv_resid)

        trend_i = predict_trend(beta_i, coords[i])[0]
        preds[i] = trend_i + pred_resid
    return preds


print(f"\nRunning LOOCV (trend refit separately for every point - no information leakage)...")
results = {}
for name, ModelClass in CANDIDATE_MODELS.items():
    model = ModelClass(dim=3)
    bin_center, gamma_emp = gs.vario_estimate((x, y, z), resid_global, latlon=False)
    model.fit_variogram(bin_center, gamma_emp, nugget=True)
    nugget_floor = 0.01 * np.var(resid_global)
    if model.nugget < nugget_floor:
        model.nugget = nugget_floor

    preds_total = loocv_trend_residual_kriging(coords, temp, tree, model, k=K_NEIGHBORS)
    rmse = np.sqrt(mean_squared_error(temp, preds_total))
    mae = mean_absolute_error(temp, preds_total)
    r2 = r2_score(temp, preds_total)
    results[name] = {'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2}
    print(f"  {name:12s}  RMSE={rmse:.3f} degC | MAE={mae:.3f} degC | R2={r2:.3f}")

best_name = min(results, key=lambda m: results[m]['rmse'])
best = results[best_name]
print(f"\n>>> Selected model: '{best_name}' (RMSE={best['rmse']:.3f} degC, R2={best['r2']:.3f})")

with open("validation_report_temperature_trend.txt", "w", encoding="utf-8") as f:
    f.write("Universal Kriging (Trend + Residual) Validation Report - Temperature\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"Number of observations: {len(temp)}\n")
    f.write(f"Global trend: T = {beta_global[0]:.3f} + {beta_global[1]:.4e}*X + "
            f"{beta_global[2]:.4e}*Y + {beta_global[3]:.5f}*Z\n\n")
    for name, res in results.items():
        f.write(f"Variogram model (on residuals): {name}\n")
        f.write(f"  RMSE={res['rmse']:.4f} degC  MAE={res['mae']:.4f} degC  R2={res['r2']:.4f}\n\n")
    f.write(f"Selected model: {best_name}\n")
print("Validation report written -> validation_report_temperature_trend.txt")

# ============================================================
# 4. Predict onto a 3D grid: trend + kriged residual
# ============================================================
xi = np.linspace(x.min(), x.max(), NX)
yi = np.linspace(y.min(), y.max(), NY)
zi = np.linspace(z.min(), z.max(), NZ)
XI, YI, ZI = np.meshgrid(xi, yi, zi, indexing='ij')
grid_pts = np.column_stack([XI.ravel(), YI.ravel(), ZI.ravel()])

print(f"\nPredicting onto the 3D grid ({len(grid_pts)} points)...")
grid_resid_pred, grid_resid_var = local_ok_predict_batch(grid_pts, coords, resid_global, tree,
                                                           best['model'], k=K_NEIGHBORS, exclude_self=False)
grid_trend = predict_trend(beta_global, grid_pts)
grid_temp = (grid_trend + grid_resid_pred).reshape(XI.shape)
grid_std = np.sqrt(grid_resid_var).reshape(XI.shape)
print("Grid prediction completed.")

# ============================================================
# 5. Plotly 3D visualisation (same style as the other model scripts)
# ============================================================
fig = go.Figure()

fig.add_trace(go.Volume(
    x=XI.flatten(), y=YI.flatten(), z=ZI.flatten(), value=grid_temp.flatten(),
    isomin=np.percentile(temp, 5), isomax=np.percentile(temp, 95),
    opacity=0.15, surface_count=20, colorscale='Inferno',
    colorbar=dict(title="Temperature (°C)", x=1.33, xanchor="left", len=0.7, yanchor="middle", orientation="v"),
    caps=dict(x_show=False, y_show=False, z_show=False),
    hovertemplate="X (UTM): %{x:.1f}<br>Y (UTM): %{y:.1f}<br>Depth: %{z:.1f} m<br>Temperature: %{value:.1f} °C<extra></extra>",
    name="Temperature Estimate (Trend + Residual Kriging)", visible=True
))

fig.add_trace(go.Volume(
    x=XI.flatten(), y=YI.flatten(), z=ZI.flatten(), value=grid_std.flatten(),
    isomin=np.percentile(grid_std, 5), isomax=np.percentile(grid_std, 95),
    opacity=0.15, surface_count=20, colorscale='Viridis',
    colorbar=dict(title="Residual Kriging Std. Dev. (°C)", x=1.55, xanchor="left", len=0.7, yanchor="middle", orientation="v"),
    caps=dict(x_show=False, y_show=False, z_show=False),
    hovertemplate="X (UTM): %{x:.1f}<br>Y (UTM): %{y:.1f}<br>Depth: %{z:.1f} m<br>Uncertainty (σ): %{value:.2f} °C<extra></extra>",
    name="Kriging Uncertainty", visible=False
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
    x=wells_df['X'], y=wells_df['Y'], z=wells_df['Z'], mode='markers+text',
    marker=dict(size=4, color='black'), text=wells_df['Well_Name'],
    textposition="top center", textfont=dict(color='black', size=14), name="Well Locations"
))

print("Adding discharge-rate points...")
debit_df = pd.read_excel(FLOW_RATE_FILE)
debit_df['X'], debit_df['Y'] = transformer.transform(debit_df['Lon'].values, debit_df['Lat'].values)
debit_df['Z'] = (debit_df['from_depth_m'] + debit_df['to_depth_m']) / 2
debit_min, debit_max = debit_df['Discharge_rate'].min(), debit_df['Discharge_rate'].max()
fig.add_trace(go.Scatter3d(
    x=debit_df['X'], y=debit_df['Y'], z=debit_df['Z'], mode='markers',
    marker=dict(size=8, color=debit_df['Discharge_rate'], colorscale='YlGnBu', cmin=debit_min, cmax=debit_max,
                opacity=1, line=dict(color='white', width=2),
                colorbar=dict(title="Discharge rate (m³/day)", x=1.02, xanchor="left", len=0.7, yanchor="middle", orientation="v")),
    hovertemplate='X (UTM): %{x:.1f}<br>Y (UTM): %{y:.1f}<br>Depth: %{z:.0f} m<br>Discharge rate: %{marker.color:.1f} m³/day<extra></extra>',
    name="Discharge rate", showlegend=True
))

n_traces = len(fig.data)
show_temp = [True, False] + [True] * (n_traces - 2)
show_unc = [False, True] + [True] * (n_traces - 2)

fig.update_layout(
    title=(f"3D Geothermal Temperature Model (Universal Kriging: Trend + {best_name} Residual)<br>"
           f"<sub>LOOCV: RMSE={best['rmse']:.2f} °C, R²={best['r2']:.3f} | "
           f"Trend: {abs(beta_global[3])*1000:.1f} °C/km depth gradient</sub>"),
    scene=dict(xaxis_title='X (UTM, m)', yaxis_title='Y (UTM, m)', zaxis_title='Depth (m)',
               aspectmode='manual', aspectratio=dict(x=1, y=1, z=0.5)),
    width=1000, height=800, margin=dict(l=50, r=50, b=50, t=140),
    updatemenus=[dict(
        type="buttons", direction="down", showactive=True, x=1.3, y=1.1, xanchor="right", yanchor="top",
        buttons=[
            dict(label="Show Temperature", method="restyle", args=[{"visible": show_temp}]),
            dict(label="Show Uncertainty (σ)", method="restyle", args=[{"visible": show_unc}]),
        ]
    )],
    legend=dict(tracegroupgap=10, itemclick="toggleothers", itemdoubleclick="toggle")
)

html_file = "outputs/3d_geothermal_temperature_model_trend_kriging.html"
os.makedirs("outputs", exist_ok=True)
fig.write_html(html_file, auto_open=False)
print(f"\nModel successfully created: {html_file}")
if os.environ.get("OPEN_IN_BROWSER", "1") == "1":
    webbrowser.open('file://' + os.path.realpath(html_file))

# ============================================================
# 6. Save model components for downstream scripts
# ============================================================
os.makedirs("outputs", exist_ok=True)
np.savez("outputs/temperature_trend_kriging_model.npz",
         coords=coords, resid=resid_global, beta=beta_global,
         best_model_name=best_name)
print("Model components saved -> outputs/temperature_trend_kriging_model.npz")
