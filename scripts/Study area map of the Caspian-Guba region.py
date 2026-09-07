# 1. İlk olaraq backend-i 'Agg' olaraq təyin edirik
import matplotlib
matplotlib.use('Agg') 

# 2. Yalnız bundan SONRA pyplot və digər kitabxanaları import edirik
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
import numpy as np
from pyproj import Geod

def add_scale_bar(ax, length_km=None, location=(0.05, 0.04), bar_height_frac=0.012):
    """
    Cartopy GeoAxes üzərinə coğrafi baxımdan dəqiq (geodezik) xətti scale bar
    (miqyas xətti) çəkir. Qara/ağ zolaqlı klassik kartoqrafik format istifadə olunur.

    ax            : scale bar əlavə olunacaq cartopy GeoAxes (PlateCarree)
    length_km     : bar-ın ümumi uzunluğu km-lə. None olarsa, xəritənin eninin
                    ~1/4-nə yaxın "yuvarlaq" bir dəyər (1/2/5/10 x 10^n) seçilir.
    location      : bar-ın sol-alt küncünün mövqeyi, hazırkı map extent-in
                    fraksiyası kimi (x_frac, y_frac), 0-1 aralığında.
    bar_height_frac: bar-ın qalınlığı, xəritənin şaquli genişliyinin fraksiyası kimi.
    """
    geod = Geod(ellps='WGS84')
    lon_min, lon_max, lat_min, lat_max = ax.get_extent(crs=ccrs.PlateCarree())
    lat_mid = (lat_min + lat_max) / 2.0

    # Xəritənin en istiqamətindəki həqiqi (geodezik) uzunluğu, orta enlikdə, km-lə
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
    # NB: bar_height birbaşa lat aralığının kiçik bir faizi kimi hesablanır
    # (əvvəlki versiyada səhvən *40 vurulurdu və bar nəhəng çıxırdı)
    bar_height = (lat_max - lat_min) * bar_height_frac

    # Ağ/qara zolaqları soldan sağa, geodezik məsafəyə görə çəkirik
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

    # Uc etiketləri: "0" və ümumi uzunluq (km)
    ax.text(x0, y0 + bar_height * 1.6, '0', ha='center', va='bottom',
            fontsize=8, transform=ccrs.PlateCarree(), zorder=11)
    ax.text(lon_cursor, y0 + bar_height * 1.6, f'{length_km:.0f} km',
            ha='center', va='bottom', fontsize=8, transform=ccrs.PlateCarree(),
            zorder=11)


# 1. Shapefile-ın oxunması
# Fayllarınızın eyni qovluqda olduğundan əmin olun
shapefile_path = "Xezeryani_Quba.shp"
gdf = gpd.read_file(shapefile_path)

# UTM Zone 39N olan məlumatları regional xəritədə düzgün göstərmək üçün WGS84 (Lat/Lon) formatına çeviririk
gdf_wgs84 = gdf.to_crs(epsg=4326)

# 2. Əsas Xəritənin (Canvas) Yaradılması
fig = plt.figure(figsize=(12, 10))

# Əsas oxlar (Xəzəryanı-Quba bölgəsi üçün) - WGS84 proyeksiyasında
ax_main = fig.add_axes([0.1, 0.1, 0.6, 0.8], projection=ccrs.PlateCarree())

# Shapefile-ı əsas xəritəyə çəkirik (narıncı rəng və naxışla)
gdf_wgs84.plot(ax=ax_main, facecolor='orange', edgecolor='black', hatch='////', linewidth=1)

# Əsas xəritənin sərhədlərini shapefile-ın ölçülərinə uyğunlaşdırırıq və bir az kənar boşluq veririk
bounds = gdf_wgs84.total_bounds  # [minx, miny, maxx, maxy]
margin = 0.05
ax_main.set_extent([bounds[0] - margin, bounds[2] + margin, 
                    bounds[1] - margin, bounds[3] + margin], crs=ccrs.PlateCarree())

# Əsas xəritə üçün grid (şəbəkə) xətləri
gl = ax_main.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Əsas xəritəyə xətti scale bar (miqyas xətti) əlavə edirik - sol-alt küncdə
add_scale_bar(ax_main, location=(0.05, 0.06))

ax_main.set_title("", fontsize=16, fontweight='bold', pad=20)

# 3. İçiçə (Inset) Xəritə: Regionun Göstərilməsi
# Xəritənin sağ tərəfində regional xəritə üçün yer ayırırıq
ax_inset = fig.add_axes([0.65, 0.45, 0.3, 0.3], projection=ccrs.PlateCarree())

# Region üçün ölkə sərhədləri, quruda və su hövzələrini əlavə edirik
ax_inset.add_feature(cfeature.LAND, facecolor='lightgray')
ax_inset.add_feature(cfeature.OCEAN, facecolor='lightblue')
ax_inset.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black', linewidth=0.5)
ax_inset.add_feature(cfeature.COASTLINE, linewidth=0.8)

# Xəritənin mərkəzini Qafqaz və Xəzər dənizi zonasına uyğunlaşdırırıq
# Təqribi koordinatlar: [Min_Lon, Max_Lon, Min_Lat, Max_Lat]
ax_inset.set_extent([38, 56, 36, 45], crs=ccrs.PlateCarree())
ax_inset.set_title("Regional map", fontsize=12, fontweight='bold')

# Şimal oxu və ya grid xətləri (İsteğe bağlı)
gl_inset = ax_inset.gridlines(draw_labels=True, linewidth=0.3, color='gray', alpha=0.5, linestyle=':')
gl_inset.top_labels = False
gl_inset.right_labels = False
gl_inset.xlabel_style = {'size': 8}
gl_inset.ylabel_style = {'size': 8}

# 4. Tədqiqat ərazisinin regional xəritədə qırmızı rəngdə vurğulanması
gdf_wgs84.plot(ax=ax_inset, facecolor='red', edgecolor='darkred', linewidth=1, zorder=5)

# Birləşdirici xətlərin çəkilməsi (Opsional - İki xəritə arasında əlaqəni göstərmək üçün)
# Bu dizaynı daha da mükəmməlləşdirmək üçün əlavə edilib
rect = Rectangle((bounds[0], bounds[1]), bounds[2]-bounds[0], bounds[3]-bounds[1],
                 linewidth=1.5, edgecolor='red', facecolor='none', transform=ccrs.PlateCarree(), zorder=6)
ax_inset.add_patch(rect)

# Nəticəni yadda saxlamaq və göstərmək
plt.savefig("Xezeryani_Quba_Xeritesi.png", dpi=300, bbox_inches='tight')
# plt.show() # Bu sətir isə mütləq şərhdə (comment) qalmalıdır