"""
Generates index.html for DSM Silt content (Limon_nv) — Oise.
Fixed 5-class silt classification, dark map, collapsible stats sidebar.
"""
import os, sys, json, io, base64, warnings
warnings.filterwarnings("ignore")

os.environ['PROJ_LIB']  = r'C:\Users\zakab\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\rasterio\proj_data'
os.environ['PROJ_DATA'] = r'C:\Users\zakab\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\rasterio\proj_data'
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling, transform_bounds
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import geopandas as gpd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FO_BASE    = os.path.join(SCRIPT_DIR, "..", "Final_outputs", "Final_outputs", "Limon_nv")
MASQUE_SHP = os.path.join(SCRIPT_DIR, "..", "data", "masque", "Departements.shp")
OUT_HTML   = os.path.join(SCRIPT_DIR, "index.html")
MAX_PX     = 750

# ── Depth definitions ─────────────────────────────────────────────
DEPTHS       = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]
DEPTH_LABEL  = {"0-5":"0–5","5-15":"5–15","15-30":"15–30",
                "30-60":"30–60","60-100":"60–100","100-200":"100–200"}
DEPTH_SUFFIX = {"0-5":"0_5cm","5-15":"5_15cm","15-30":"15_30cm",
                "30-60":"30_60cm","60-100":"60_100cm","100-200":"100_200cm"}
DEPTH_FOLDER = {"0-5":"0-5","5-15":"5-15","15-30":"15-30",
                "30-60":"30-60","60-100":"60-100","100-200":"100-200"}

# ── Fixed silt classification (5 classes, gradient colormap) ──────
SILT_LABELS = ['≤ 250', '250 – 450', '450 – 600', '600 – 750', '≥ 750']

# Sequential gradient: blue → purple (distinct from clay YlOrBr and sand YlGnBu)
CMAP_SILT = plt.get_cmap('BuPu')
NORM_SILT  = mcolors.Normalize(vmin=250, vmax=750, clip=True)

# Sample legend swatch colors at each class midpoint
_mids = [250.0, 350.0, 525.0, 675.0, 750.0]
def _silt_hex(v):
    return mcolors.to_hex(CMAP_SILT(NORM_SILT(v))[:3])

FIXED_CLASSES = [
    {"min": -999, "max": 250,  "color": _silt_hex(_mids[0]), "label": SILT_LABELS[0]},
    {"min": 250,  "max": 450,  "color": _silt_hex(_mids[1]), "label": SILT_LABELS[1]},
    {"min": 450,  "max": 600,  "color": _silt_hex(_mids[2]), "label": SILT_LABELS[2]},
    {"min": 600,  "max": 750,  "color": _silt_hex(_mids[3]), "label": SILT_LABELS[3]},
    {"min": 750,  "max": 9999, "color": _silt_hex(_mids[4]), "label": SILT_LABELS[4]},
]

# ── Load study-area mask (Oise) ───────────────────────────────────
print("Loading study-area mask...")
masque_wgs84 = gpd.read_file(MASQUE_SHP).to_crs("EPSG:4326")
masque_union = masque_wgs84.union_all() if hasattr(masque_wgs84, 'union_all') \
               else masque_wgs84.unary_union
print(f"  Mask loaded — {masque_wgs84.shape[0]} polygon(s)")

def clip_pts_to_mask(gdf):
    return gdf[gdf.geometry.within(masque_union)].copy()

def find_file(depth_dir, *candidates):
    """Return the first existing path among candidates."""
    for name in candidates:
        p = os.path.join(depth_dir, name)
        if os.path.exists(p):
            return p
    return None

def shp_to_geojson(shp_path):
    gdf = gpd.read_file(shp_path).to_crs("EPSG:4326")
    gdf = clip_pts_to_mask(gdf)
    val_col = [c for c in gdf.columns if c not in ("id","geometry")][0]
    features = []
    for _, row in gdf.iterrows():
        if row.geometry is None: continue
        v = float(row[val_col]) if pd.notna(row[val_col]) else None
        features.append({
            "type": "Feature",
            "geometry": {"type":"Point",
                         "coordinates":[round(row.geometry.x,5),
                                        round(row.geometry.y,5)]},
            "properties": {"value": round(v,1) if v is not None else None},
        })
    return {"type":"FeatureCollection","features":features}, len(features)

# ── Process each depth ────────────────────────────────────────────
all_data = {}

for depth in DEPTHS:
    suffix    = DEPTH_SUFFIX[depth]
    folder    = DEPTH_FOLDER[depth]
    depth_dir = os.path.join(FO_BASE, folder)
    print(f"\nDepth {depth} cm ...")

    entry = {"img":None,"bounds":None,"center":[49.5,2.4],
             "classes": FIXED_CLASSES,"pts":None,"imp":{},"met":{},
             "modelable": depth != "100-200"}

    # ── RF Mean map ───────────────────────────────────────────────
    tif_path = find_file(depth_dir,
                         f"carte_RF_moyenne_limon_nv_g_kg_{suffix}.tif",
                         f"carte_RF_moyenne_limon_g_kg_{suffix}.tif")
    if tif_path:
        with rasterio.open(tif_path) as src:
            t4326,w,h = calculate_default_transform(
                src.crs,"EPSG:4326",src.width,src.height,*src.bounds)
            data = np.full((h,w),np.nan,dtype="float32")
            reproject(source=rasterio.band(src,1),destination=data,
                      src_transform=src.transform,src_crs=src.crs,
                      dst_transform=t4326,dst_crs="EPSG:4326",
                      resampling=Resampling.bilinear)
            nodata = src.nodata
            data   = np.where(data == nodata, np.nan, data) if nodata is not None else data
            bnds   = transform_bounds(src.crs,"EPSG:4326",*src.bounds)

        # Apply continuous gradient silt colormap
        rgba = CMAP_SILT(NORM_SILT(np.where(np.isnan(data), 0, data)))
        rgba = rgba.astype(float)
        rgba[...,3] = np.where(np.isnan(data), 0, 0.88)
        pil  = Image.fromarray((rgba*255).astype(np.uint8),"RGBA")
        if pil.width > MAX_PX:
            pil = pil.resize((MAX_PX, int(pil.height*MAX_PX/pil.width)), Image.LANCZOS)
        buf = io.BytesIO(); pil.save(buf,"PNG"); buf.seek(0)
        entry["img"]    = base64.b64encode(buf.read()).decode()
        entry["bounds"] = [[bnds[1],bnds[0]],[bnds[3],bnds[2]]]
        entry["center"] = [(bnds[1]+bnds[3])/2,(bnds[0]+bnds[2])/2]
        _flat = data[::12, ::12].flatten()
        _flat = _flat[~np.isnan(_flat)]
        if len(_flat) > 1500:
            rng = np.random.default_rng(42)
            _flat = rng.choice(_flat, 1500, replace=False)
        entry["raster_vals"] = [round(float(v), 1) for v in sorted(_flat)]
        print(f"  Map OK — {len(entry['img'])//1024} KB")
    else:
        print(f"  Map not found for depth {depth}")

    # ── Sample points ─────────────────────────────────────────────
    shp_path = os.path.join(depth_dir, f"points_apprentissage_limon_nv_g_kg_{suffix}.shp")
    if os.path.exists(shp_path):
        try:
            geojson, n = shp_to_geojson(shp_path)
            entry["pts"] = geojson
            print(f"  Sample points OK — {n} pts")
        except Exception as e:
            print(f"  Sample points error: {e}")
    else:
        print(f"  Sample points not found: {shp_path}")

    # ── MDI importance ────────────────────────────────────────────
    imp_path = find_file(depth_dir,
                         f"importance_MDI_50iter_redistrib_limon_nv_g_kg_{suffix}.xlsx",
                         f"importance_MDI_50iter_redistrib_limon_g_kg_{suffix}.xlsx")
    if imp_path:
        df_imp = pd.read_excel(imp_path)
        order  = (df_imp.groupby("variable")["importance"]
                  .median().sort_values(ascending=False).index.tolist())
        for var in order:
            entry["imp"][var] = df_imp.loc[
                df_imp["variable"]==var,"importance"].round(3).tolist()
        print(f"  MDI OK — {len(entry['imp'])} variables")

    # ── Validation metrics ─────────────────────────────────────────
    met_path = find_file(depth_dir,
                         f"metriques_iterations_limon_nv_g_kg_{suffix}.xlsx",
                         f"metriques_iterations_limon_g_kg_{suffix}.xlsx")
    if met_path:
        df_met = pd.read_excel(met_path)
        entry["met"]["r2"]   = df_met["r2"].dropna().round(4).tolist()
        entry["met"]["rmse"] = df_met["rmse_gkg"].dropna().round(1).tolist()
        print(f"  Metrics OK — {len(entry['met']['r2'])} iterations")

    all_data[depth] = entry

# ── Build HTML ────────────────────────────────────────────────────
data_js    = json.dumps(all_data, ensure_ascii=False)
depths_js  = json.dumps(DEPTHS)
dept_js    = json.dumps(json.loads(masque_wgs84.to_json()), ensure_ascii=False)
classes_js = json.dumps(FIXED_CLASSES, ensure_ascii=False)
print(f"  Dept GeoJSON — {len(dept_js)//1024} KB")

tab_btns = "\n    ".join(
    f'<button class="dtab{" active" if d=="0-5" else ""}" '
    f'onclick="showDepth(\'{d}\',this)">{DEPTH_LABEL[d]} cm</button>'
    for d in DEPTHS
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DSM — Silt Content · Oise</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{
  height:100%;overflow:hidden;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  background:#0a0a12;
}}
#header{{
  position:fixed;top:0;left:0;right:0;z-index:1100;
  height:54px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 22px 0 24px;
  background:rgba(6,8,18,0.92);
  backdrop-filter:blur(10px);
  border-bottom:1px solid rgba(255,255,255,0.07);
  box-shadow:0 2px 16px rgba(0,0,0,0.55);
}}
#header-left h1{{font-size:0.92rem;font-weight:700;color:#e2e8f0;letter-spacing:0.3px;}}
#header-left p{{font-size:0.68rem;color:rgba(255,255,255,0.38);margin-top:1px;}}
.depth-tabs{{display:flex;align-items:center;gap:5px}}
.dtab{{
  padding:6px 17px;border-radius:20px;
  border:1.5px solid rgba(255,255,255,0.10);
  background:rgba(255,255,255,0.05);
  color:rgba(255,255,255,0.52);
  font-size:0.78rem;font-weight:500;cursor:pointer;
  transition:background .15s,color .15s,border-color .15s;
  white-space:nowrap;line-height:1;
}}
.dtab:hover{{background:rgba(255,255,255,0.11);color:rgba(255,255,255,0.85);border-color:rgba(255,255,255,0.22);}}
.dtab.active{{
  background:linear-gradient(135deg,#2563eb,#3b82f6);
  border-color:#3b82f6;color:#fff;font-weight:700;
  box-shadow:0 0 12px rgba(59,130,246,0.45);
}}
#map{{position:fixed;top:54px;left:0;right:0;bottom:0;}}
#legend-panel{{
  position:fixed;bottom:24px;right:18px;z-index:1050;
  background:rgba(6,8,18,0.90);backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,0.09);border-radius:12px;
  padding:13px 16px 11px;min-width:185px;max-width:220px;
  box-shadow:0 6px 28px rgba(0,0,0,0.65);color:#dde6f0;font-size:0.78rem;
}}
.leg-title{{font-size:0.67rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.9px;color:rgba(255,255,255,0.38);margin-bottom:9px;}}
.leg-item{{display:flex;align-items:center;gap:9px;padding:2.5px 0;}}
.leg-swatch{{width:20px;height:11px;border-radius:3px;
  border:1px solid rgba(255,255,255,0.12);flex-shrink:0;}}
.leg-sep{{height:1px;background:rgba(255,255,255,0.08);margin:9px 0 7px;}}
.leg-pts{{
  display:flex;align-items:center;gap:9px;
  padding:5px 7px;margin:0 -7px;border-radius:7px;
  cursor:pointer;transition:background .15s;user-select:none;
}}
.leg-pts:hover{{background:rgba(255,255,255,0.07)}}
.leg-pts .pt-icon{{
  width:12px;height:12px;border-radius:50%;
  background:#38bdf8;border:1px solid rgba(255,255,255,0.25);
  flex-shrink:0;transition:opacity .2s;
}}
.leg-pts .raster-icon{{
  width:20px;height:11px;border-radius:3px;
  border:1px solid rgba(255,255,255,0.12);flex-shrink:0;transition:opacity .2s;
}}
.leg-pts .line-icon{{
  width:22px;height:3px;border-radius:2px;
  background:rgba(255,255,255,0.85);flex-shrink:0;transition:opacity .2s;
}}
.leg-pts .pt-label{{transition:opacity .2s;}}
.leg-pts.pts-off .pt-icon,
.leg-pts.pts-off .raster-icon,
.leg-pts.pts-off .line-icon{{opacity:0.22;}}
.leg-pts.pts-off .pt-label{{opacity:0.35;text-decoration:line-through;}}
.check-icon{{margin-left:auto;font-size:0.7rem;color:rgba(255,255,255,0.35);transition:opacity .2s;}}
.pts-off .check-icon{{opacity:0;}}
.line-icon{{width:22px;height:3px;border-radius:2px;background:rgba(255,255,255,0.85);flex-shrink:0;}}
.raster-icon{{width:20px;height:11px;border-radius:3px;border:1px solid rgba(255,255,255,0.12);flex-shrink:0;}}
#sidebar{{
  position:fixed;top:54px;left:0;bottom:0;z-index:1080;
  width:375px;display:flex;flex-direction:row;
  transform:translateX(calc(-100% + 28px));
  transition:transform .32s cubic-bezier(.4,0,.2,1);
}}
#sidebar.open{{transform:translateX(0);}}
#sb-content{{
  flex:1;min-width:0;overflow-y:auto;
  background:rgba(6,8,18,0.93);backdrop-filter:blur(14px);
  border-right:1px solid rgba(255,255,255,0.08);
  padding:14px 13px 28px;display:flex;flex-direction:column;gap:16px;
}}
#sb-content::-webkit-scrollbar{{width:4px;}}
#sb-content::-webkit-scrollbar-track{{background:transparent;}}
#sb-content::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.1);border-radius:2px;}}
#sb-tab{{
  flex-shrink:0;width:28px;
  display:flex;flex-direction:column;align-items:center;justify-content:center;
  cursor:pointer;background:rgba(6,8,18,0.92);
  border:1px solid rgba(255,255,255,0.09);border-left:none;
  border-radius:0 10px 10px 0;
  box-shadow:4px 0 18px rgba(0,0,0,0.55);
  height:64px;margin:auto 0;user-select:none;transition:background .15s;
}}
#sb-tab:hover{{background:rgba(20,24,48,0.97);}}
.hamburger{{display:flex;flex-direction:column;align-items:center;gap:4px;}}
.hamburger span{{display:block;width:14px;height:2px;background:rgba(255,255,255,0.58);border-radius:1px;transition:background .15s;}}
#sb-tab:hover .hamburger span{{background:rgba(255,255,255,0.88);}}
.sb-section{{display:flex;flex-direction:column;gap:6px;}}
.sb-title{{font-size:0.67rem;font-weight:700;text-transform:uppercase;
  letter-spacing:0.8px;color:rgba(255,255,255,0.32);
  padding-bottom:7px;border-bottom:1px solid rgba(255,255,255,0.07);}}
.sb-no-data{{color:rgba(255,255,255,0.25);font-size:0.78rem;padding:20px 0;text-align:center;}}
.sb-title-row{{display:flex;align-items:center;justify-content:space-between;
  padding-bottom:7px;border-bottom:1px solid rgba(255,255,255,0.07);}}
.sb-title-row .sb-title{{padding-bottom:0;border-bottom:none;}}
.info-btn{{
  background:none;border:1px solid rgba(255,255,255,0.18);border-radius:50%;
  color:rgba(255,255,255,0.45);font-size:0.72rem;line-height:1;
  width:17px;height:17px;cursor:pointer;display:flex;align-items:center;
  justify-content:center;flex-shrink:0;transition:color .15s,border-color .15s;padding:0;
}}
.info-btn:hover{{color:#93c5fd;border-color:#93c5fd;}}
.info-btn.active{{color:#38bdf8;border-color:#38bdf8;}}
#var-info-panel{{display:none;overflow-y:auto;max-height:260px;font-size:0.74rem;color:#cbd5e1;}}
#var-info-panel::-webkit-scrollbar{{width:3px;}}
#var-info-panel::-webkit-scrollbar-thumb{{background:rgba(255,255,255,0.1);border-radius:2px;}}
.vi-row{{display:flex;gap:8px;padding:3.5px 0;border-bottom:1px solid rgba(255,255,255,0.04);}}
.vi-label{{font-weight:600;color:#93c5fd;white-space:nowrap;min-width:120px;font-size:0.72rem;}}
.vi-desc{{color:rgba(255,255,255,0.5);font-size:0.71rem;line-height:1.4;}}
.leaflet-control-zoom a{{
  background:rgba(6,8,18,0.88)!important;color:rgba(255,255,255,0.65)!important;
  border-color:rgba(255,255,255,0.10)!important;}}
.leaflet-control-zoom a:hover{{background:rgba(30,35,60,0.95)!important;color:#fff!important;}}
.leaflet-control-attribution{{background:rgba(0,0,0,0.45)!important;color:rgba(255,255,255,0.28)!important;font-size:9px;}}
.leaflet-control-attribution a{{color:rgba(255,255,255,0.35)!important}}
.pt-tip{{
  background:rgba(6,8,18,0.93);border:1px solid rgba(255,255,255,0.13);
  border-radius:7px;padding:5px 11px;font-size:12px;color:#e2e8f0;
  box-shadow:0 4px 14px rgba(0,0,0,0.6);
}}
#no-model-badge{{
  position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);
  z-index:1060;background:rgba(6,8,18,0.88);
  border:1px solid rgba(255,255,255,0.09);border-radius:14px;
  padding:28px 36px;text-align:center;color:#94a3b8;display:none;
}}
#no-model-badge h3{{color:#cbd5e1;font-size:1rem;margin-bottom:8px}}
#no-model-badge p{{font-size:0.82rem;line-height:1.6}}
</style>
</head>
<body>

<div id="header">
  <div id="header-left">
    <h1>Digital Soil Mapping — Silt Content</h1>
    <p>Oise Department, France &nbsp;·&nbsp; 30 m resolution</p>
  </div>
  <div class="depth-tabs">
    {tab_btns}
  </div>
</div>

<div id="map"></div>

<div id="legend-panel">
  <div class="leg-title">Silt content (g/kg)</div>
  <div id="leg-classes"></div>
  <div class="leg-sep"></div>
  <div class="leg-pts" id="raster-toggle" onclick="toggleRaster()">
    <div class="raster-icon" id="raster-icon"></div>
    <span class="pt-label" id="raster-label">Silt map</span>
    <span class="check-icon">&#10003;</span>
  </div>
  <div class="leg-pts" id="pts-toggle" onclick="togglePoints()">
    <div class="pt-icon"></div>
    <span class="pt-label">Sample points</span>
    <span class="check-icon">&#10003;</span>
  </div>
  <div class="leg-pts" id="dept-toggle" onclick="toggleDept()">
    <div class="line-icon"></div>
    <span class="pt-label">Oise boundary</span>
    <span class="check-icon">&#10003;</span>
  </div>
</div>

<div id="no-model-badge">
  <h3>Depth not modelled</h3>
  <p id="no-model-txt"></p>
</div>

<div id="sidebar">
  <div id="sb-content">
    <div class="sb-section">
      <div class="sb-title-row">
        <span class="sb-title">Covariate importance (MDI — 50 iter.)</span>
        <button class="info-btn" id="info-btn" onclick="toggleVarInfo()" title="Covariate descriptions">ⓘ</button>
      </div>
      <div id="var-info-panel"></div>
      <div id="chart-imp"></div>
    </div>
    <div class="sb-section">
      <div class="sb-title">Validation metrics — 50 iterations</div>
      <div id="chart-met"></div>
    </div>
    <div class="sb-section">
      <div class="sb-title">Map value distribution</div>
      <div id="chart-raster"></div>
    </div>
  </div>
  <div id="sb-tab" onclick="toggleSidebar()">
    <div class="hamburger"><span></span><span></span><span></span></div>
  </div>
</div>

<script>
const DATA      = {data_js};
const DEPTHS    = {depths_js};
const DEPT_GJ   = {dept_js};
const CLASSES   = {classes_js};

let rfLayer = null, ptLayer = null, deptLayer = null;
let rfOn = true, ptsOn = true, deptOn = true;
let sidebarOpen = false, currentDepth = DEPTHS[0];

const map = L.map('map', {{zoomControl:true, attributionControl:true, minZoom:8, maxZoom:17}});
L.tileLayer(
  'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png',
  {{attribution:'&copy; <a href="https://www.openstreetmap.org">OpenStreetMap</a> &copy; <a href="https://carto.com">CARTO</a>',
   maxZoom:17, minZoom:8, subdomains:'abcd'}}
).addTo(map);
map.setView([49.5, 2.4], 9);

deptLayer = L.geoJSON(DEPT_GJ, {{
  style:{{color:'rgba(255,255,255,0.85)',weight:2.2,dashArray:'6 4',fillOpacity:0,opacity:1}}
}}).addTo(map);

(function buildStaticLegend() {{
  const el = document.getElementById('leg-classes');
  el.innerHTML = CLASSES.map(c =>
    `<div class="leg-item">
       <div class="leg-swatch" style="background:${{c.color}}"></div>
       <span>${{c.label}}</span>
     </div>`
  ).join('');
  document.getElementById('raster-icon').style.background =
    `linear-gradient(to right,${{CLASSES[0].color}},${{CLASSES[CLASSES.length-1].color}})`;
}})();

const depthLabels = {{"0-5":"0–5","5-15":"5–15","15-30":"15–30","30-60":"30–60","60-100":"60–100","100-200":"100–200"}};

function showDepth(d, btn) {{
  document.querySelectorAll('.dtab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  currentDepth = d;

  if (varInfoVisible) {{
    varInfoVisible = false;
    document.getElementById('var-info-panel').style.display = 'none';
    document.getElementById('chart-imp').style.display = 'block';
    document.getElementById('info-btn').classList.remove('active');
  }}

  const info = DATA[d];
  if (rfLayer)  {{ map.removeLayer(rfLayer);  rfLayer  = null; }}
  if (ptLayer)  {{ map.removeLayer(ptLayer);  ptLayer  = null; }}

  const badge    = document.getElementById('no-model-badge');
  const legPanel = document.getElementById('legend-panel');

  if (!info.modelable) {{
    const n = info.pts ? info.pts.features.length : 0;
    document.getElementById('no-model-txt').innerHTML =
      `Only <b>${{n}}</b> sample point${{n!==1?'s':''}} available at this depth.`;
    badge.style.display = 'block';
    legPanel.style.display = 'none';
    map.setView([49.5,2.4], 9);
    return;
  }}
  badge.style.display = 'none';
  legPanel.style.display = 'block';

  document.getElementById('raster-label').textContent = `Silt map (${{depthLabels[d]}} cm)`;

  if (info.img && info.bounds) {{
    rfLayer = L.imageOverlay('data:image/png;base64,'+info.img, info.bounds, {{opacity:0.88, interactive:false}});
    if (rfOn) rfLayer.addTo(map);
    map.fitBounds(info.bounds, {{paddingTopLeft:[0,8], paddingBottomRight:[0,8]}});
  }} else {{
    map.setView(info.center || [49.5,2.4], 9);
  }}

  if (info.pts && info.pts.features.length > 0) {{
    ptLayer = L.geoJSON(info.pts, {{
      pointToLayer:(f,ll) => L.circleMarker(ll, {{
        radius:3, fillColor:'#38bdf8',
        color:'rgba(255,255,255,0.25)', weight:0.8,
        opacity:1, fillOpacity:0.85,
      }}),
      onEachFeature:(f,layer) => {{
        const v = f.properties.value;
        layer.bindTooltip(
          v !== null ? `<b>Silt:</b> ${{v.toFixed(1)}} g/kg` : '<i>No value</i>',
          {{direction:'top', offset:[0,-4], className:'pt-tip'}}
        );
      }}
    }});
    if (ptsOn) ptLayer.addTo(map);
  }}

  if (deptLayer) deptLayer.bringToFront();
  if (sidebarOpen) renderCharts(d);
}}

function toggleRaster() {{
  rfOn = !rfOn;
  document.getElementById('raster-toggle').classList.toggle('pts-off', !rfOn);
  if (rfLayer) {{ if (rfOn) rfLayer.addTo(map); else map.removeLayer(rfLayer); }}
}}
function togglePoints() {{
  ptsOn = !ptsOn;
  document.getElementById('pts-toggle').classList.toggle('pts-off', !ptsOn);
  if (ptLayer) {{ if (ptsOn) ptLayer.addTo(map); else map.removeLayer(ptLayer); }}
}}
function toggleDept() {{
  deptOn = !deptOn;
  document.getElementById('dept-toggle').classList.toggle('pts-off', !deptOn);
  if (deptLayer) {{ if (deptOn) deptLayer.addTo(map); else map.removeLayer(deptLayer); }}
}}

function toggleSidebar() {{
  sidebarOpen = !sidebarOpen;
  document.getElementById('sidebar').classList.toggle('open', sidebarOpen);
  if (sidebarOpen) renderCharts(currentDepth);
}}

const VAR_INFO = {{
  'Altitude':             {{ label:'Elevation',             desc:'Digital elevation model' }},
  'Aspect':               {{ label:'Aspect',                desc:'Slope orientation' }},
  'BI':                   {{ label:'Brightness Index',      desc:'Soil brightness from satellite reflectance' }},
  'BSI':                  {{ label:'Bare Soil Index',       desc:'Bare soil spectral index' }},
  'CI':                   {{ label:'Color Index',           desc:'Spectral color index from satellite imagery' }},
  'Courbure':             {{ label:'Curvature',             desc:'Terrain profile curvature' }},
  'EVI':                  {{ label:'EVI',                   desc:'Enhanced Vegetation Index' }},
  'HPPR':                 {{ label:'HPPR',                  desc:'Height above the nearest drainage network' }},
  'LSWI':                 {{ label:'LSWI',                  desc:'Land Surface Water Index' }},
  'MRVBF':                {{ label:'MRVBF',                 desc:'Multi-Resolution Valley Bottom Flatness' }},
  'NDVI':                 {{ label:'NDVI',                  desc:'Normalized Difference Vegetation Index' }},
  'NDWI':                 {{ label:'NDWI',                  desc:'Normalized Difference Water Index' }},
  'OCS':                  {{ label:'Land cover',            desc:'Soil occupation / land use map' }},
  'Pente':                {{ label:'Slope',                 desc:'Terrain slope' }},
  'Prairies_permanentes': {{ label:'Permanent grasslands',  desc:'Binary: 1 = grassland, 0 = other land' }},
  'Rugosite_etendue':     {{ label:'Extended roughness',    desc:'Multi-scale terrain roughness index' }},
  'Rugosite_std':         {{ label:'Roughness (std)',       desc:'Standard deviation of terrain roughness' }},
  'TPI':                  {{ label:'TPI',                   desc:'Topographic Position Index' }},
  'TWI':                  {{ label:'TWI',                   desc:'Topographic Wetness Index' }},
  'geologie':             {{ label:'Geology',               desc:'Lithological and geological map' }},
  'precip_annuelle':      {{ label:'Annual precipitation',  desc:'Mean annual precipitation' }},
  'sols_rrp':             {{ label:'Reference soil map',    desc:'Regional Pedological Reference map' }},
  'temp_moy_annuelle':    {{ label:'Mean annual temp.',     desc:'Mean annual air temperature' }},
}};

let varInfoVisible = false;
function toggleVarInfo() {{
  varInfoVisible = !varInfoVisible;
  const panel = document.getElementById('var-info-panel');
  const impEl = document.getElementById('chart-imp');
  const btn   = document.getElementById('info-btn');
  btn.classList.toggle('active', varInfoVisible);
  if (varInfoVisible) {{
    const vars = Object.keys(DATA[currentDepth]?.imp || {{}});
    panel.innerHTML = vars.map(v => {{
      const info = VAR_INFO[v] || {{ label: v, desc: '—' }};
      return `<div class="vi-row"><span class="vi-label">${{info.label}}</span><span class="vi-desc">${{info.desc}}</span></div>`;
    }}).join('');
    panel.style.display = 'block';
    impEl.style.display  = 'none';
  }} else {{
    panel.style.display = 'none';
    impEl.style.display  = 'block';
  }}
}}

function renderCharts(d) {{
  const info   = DATA[d];
  const impEl  = document.getElementById('chart-imp');
  const metEl  = document.getElementById('chart-met');

  if (!info.modelable || !Object.keys(info.imp).length) {{
    impEl.innerHTML = '<div class="sb-no-data">Not modelled at this depth.</div>';
    metEl.innerHTML = '';
    return;
  }}

  const imp     = info.imp;
  const vars    = Object.keys(imp);
  const varsAsc = [...vars].reverse();

  Plotly.react(impEl,
    varsAsc.map(v => {{
      const lbl = (VAR_INFO[v] || {{}}).label || v;
      return {{
      x:imp[v], name:lbl.length>22?lbl.slice(0,20)+'…':lbl,
      type:'box', orientation:'h', boxpoints:'outliers',
      marker:{{color:'rgba(96,165,250,0.55)',size:3.5}},
      line:{{color:'#93c5fd',width:1.3}},
      fillcolor:'rgba(59,130,246,0.12)',
      whiskerwidth:0.45, showlegend:false,
      }}; }}),
    {{
      height:Math.max(270, vars.length*25+55),
      margin:{{l:130,r:12,t:8,b:40}},
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(255,255,255,0.015)',
      xaxis:{{title:{{text:'Importance (%)',font:{{color:'rgba(255,255,255,0.36)',size:10}}}},
        color:'rgba(255,255,255,0.32)',gridcolor:'rgba(255,255,255,0.06)',
        zeroline:true,zerolinecolor:'rgba(255,255,255,0.10)',
        tickfont:{{size:9,color:'rgba(255,255,255,0.42)'}}}},
      yaxis:{{tickfont:{{size:9,color:'rgba(255,255,255,0.62)'}},automargin:true,color:'rgba(255,255,255,0.32)'}},
      hovermode:false,
    }},
    {{displayModeBar:false, responsive:true}}
  );

  const met   = info.met;
  const r2s   = met.r2   || [];
  const rmses = met.rmse || [];
  if (r2s.length === 0) {{
    metEl.innerHTML = '<div class="sb-no-data">No validation metrics available.</div>';
    return;
  }}
  const maxR2    = Math.max(...r2s).toFixed(3);
  const minR2    = Math.min(...r2s).toFixed(3);
  const meanR2   = (r2s.reduce((a,b)=>a+b,0)/r2s.length).toFixed(3);
  const maxRmse  = Math.max(...rmses).toFixed(1);
  const minRmse  = Math.min(...rmses).toFixed(1);
  const meanRmse = (rmses.reduce((a,b)=>a+b,0)/rmses.length).toFixed(1);

  Plotly.react(metEl,
    [
      {{y:r2s, name:'R²', type:'box', boxpoints:'all', jitter:0.42, pointpos:0,
        marker:{{color:'rgba(52,211,153,0.48)',size:4}},
        line:{{color:'#6ee7b7',width:1.8}}, fillcolor:'rgba(52,211,153,0.11)',showlegend:false}},
      {{y:rmses, name:'RMSE', type:'box', boxpoints:'all', jitter:0.42, pointpos:0,
        xaxis:'x2', yaxis:'y2',
        marker:{{color:'rgba(251,146,60,0.48)',size:4}},
        line:{{color:'#fdba74',width:1.8}}, fillcolor:'rgba(251,146,60,0.11)',showlegend:false}},
    ],
    {{
      height:260, margin:{{l:38,r:38,t:66,b:30}},
      grid:{{rows:1,columns:2,pattern:'independent'}},
      paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(255,255,255,0.015)',
      xaxis:{{domain:[0,0.43],title:{{text:'R²',font:{{color:'#6ee7b7',size:11}}}},
        tickfont:{{size:9,color:'rgba(255,255,255,0.42)'}},color:'rgba(255,255,255,0.22)',gridcolor:'rgba(0,0,0,0)'}},
      xaxis2:{{domain:[0.57,1],title:{{text:'RMSE (g/kg)',font:{{color:'#fdba74',size:11}}}},
        tickfont:{{size:9,color:'rgba(255,255,255,0.42)'}},color:'rgba(255,255,255,0.22)',gridcolor:'rgba(0,0,0,0)'}},
      yaxis:{{range:[0,1],gridcolor:'rgba(255,255,255,0.07)',zeroline:false,
        tickfont:{{size:9,color:'rgba(255,255,255,0.42)'}},color:'rgba(255,255,255,0.22)'}},
      yaxis2:{{gridcolor:'rgba(255,255,255,0.07)',zeroline:false,
        tickfont:{{size:9,color:'rgba(255,255,255,0.42)'}},color:'rgba(255,255,255,0.22)'}},
      shapes:[{{type:'line',xref:'x',yref:'y',x0:-0.45,x1:0.45,y0:0.5,y1:0.5,
        line:{{color:'rgba(239,68,68,0.65)',width:1.4,dash:'dot'}}}}],
      annotations:[
        {{xref:'paper',yref:'paper',x:0.215,y:1.01,
          text:`Max: ${{maxR2}}<br>Mean: ${{meanR2}}<br>Min: ${{minR2}}`,
          showarrow:false,font:{{color:'#6ee7b7',size:9.5}},xanchor:'center',yanchor:'bottom',align:'center'}},
        {{xref:'paper',yref:'paper',x:0.785,y:1.01,
          text:`Max: ${{maxRmse}}<br>Mean: ${{meanRmse}}<br>Min: ${{minRmse}} g/kg`,
          showarrow:false,font:{{color:'#fdba74',size:9.5}},xanchor:'center',yanchor:'bottom',align:'center'}},
        {{xref:'paper',yref:'y',x:0.43,y:0.5,text:'0.5',showarrow:false,
          font:{{color:'rgba(239,68,68,0.65)',size:9}},xanchor:'right',yanchor:'bottom'}},
      ],
      hovermode:false,
    }},
    {{displayModeBar:false, responsive:true}}
  );

  // ── Raster value distribution ─────────────────────────────
  const rastEl = document.getElementById('chart-raster');
  const rv     = info.raster_vals || [];
  if (rv.length === 0) {{
    rastEl.innerHTML = '<div class="sb-no-data">No raster data at this depth.</div>';
  }} else {{
    const maxV  = Math.max(...rv).toFixed(1);
    const minV  = Math.min(...rv).toFixed(1);
    const meanV = (rv.reduce((a,b)=>a+b,0)/rv.length).toFixed(1);
    Plotly.react(rastEl,
      [{{
        x:rv, type:'box', orientation:'h', boxpoints:'outliers',
        marker:{{color:'rgba(251,191,36,0.45)',size:3}},
        line:{{color:'#fbbf24',width:1.6}},
        fillcolor:'rgba(251,191,36,0.10)',
        whiskerwidth:0.5, showlegend:false, name:'',
      }}],
      {{
        height:130, margin:{{l:10,r:12,t:40,b:28}},
        paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(255,255,255,0.015)',
        xaxis:{{
          title:{{text:'Silt (g/kg)',font:{{color:'rgba(255,255,255,0.36)',size:10}}}},
          color:'rgba(255,255,255,0.32)',gridcolor:'rgba(255,255,255,0.06)',
          tickfont:{{size:9,color:'rgba(255,255,255,0.42)'}},zeroline:false,
        }},
        yaxis:{{visible:false}},
        annotations:[{{
          xref:'paper',yref:'paper',x:0.5,y:1.08,
          text:`Max: ${{maxV}} · Mean: ${{meanV}} · Min: ${{minV}} g/kg`,
          showarrow:false,font:{{color:'#fbbf24',size:9}},xanchor:'center',yanchor:'bottom',
        }}],
        hovermode:false,
      }},
      {{displayModeBar:false, responsive:true}}
    );
  }}
}}

showDepth(DEPTHS[0], document.querySelector('.dtab.active'));
setTimeout(() => map.invalidateSize(), 80);
</script>
</body>
</html>"""

with open(OUT_HTML, "w", encoding="utf-8") as f:
    f.write(html)
print(f"\nDone — {OUT_HTML} ({os.path.getsize(OUT_HTML)//1024} KB)")
