"""
UrbanSense - Bengaluru Heat Lab (Streamlit front end for the frozen ConvLSTM LST/UHI model).

Everything numeric shown here is read from files written by the model repo.
Nothing is estimated, scaled or invented in the UI. Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import folium
import numpy as np
import streamlit as st
from branca.colormap import LinearColormap
from folium.plugins import Draw
from shapely.geometry import shape
from shapely.ops import transform as shp_transform
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from services import geocode, raster_io, uhi_backend as be  # noqa: E402

st.set_page_config(page_title="UrbanSense - Bengaluru Heat Lab", page_icon="🌡️", layout="wide")
st.markdown(f"<style>{(Path(__file__).parent / 'ui' / 'styles.css').read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True)

ss = st.session_state
ss.setdefault("lat", 12.9716)
ss.setdefault("lon", 77.5946)
ss.setdefault("result", None)
ss.setdefault("result_meta", None)
ss.setdefault("drawn_polygon", None)      # the area to treat, kept across reruns
ss.setdefault("last_click", None)         # last map click already handled
ss.setdefault("map_nonce", 0)             # bump = fresh map with an empty drawing layer (one polygon only)
ss.setdefault("map_center", [12.9716, 77.5946])  # only changes on search, so the map is not rebuilt


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def cached_overlay(path: str, mtime: float, vmin: float, vmax: float, cmap: str):
    return raster_io.overlay(Path(path), vmin, vmax, cmap)


@st.cache_data(show_spinner=False)
def cached_range(path: str, mtime: float, symmetric: bool):
    return raster_io.value_range(Path(path), symmetric)


def mt(p: Path) -> float:
    return p.stat().st_mtime


def polygon_area_km2(geojson: dict) -> float:
    geom = shp_transform(be._to_grid.transform, shape(geojson))
    return geom.area / 1e6


def category(uhi: float) -> tuple[str, str]:
    """Display bands only (my suggestion, not a model output). Model UHI error is ~1 C, so bands are coarse."""
    if not np.isfinite(uhi):
        return "No data", "pill-warn"
    if uhi >= 2:
        return "Strong UHI", "pill-high"
    if uhi >= 1:
        return "Moderate UHI", "pill-warn"
    if uhi >= 0:
        return "Weak / none", "pill-model"
    return "Cooler than rural reference", "pill-model"


# ---------------------------------------------------------------- header
h1, h2, h3 = st.columns([2.2, 1.4, 1.6])
with h1:
    st.markdown('<div class="brand">Urban<span>°</span>Sense</div><div class="eyebrow">Bengaluru heat lab</div>',
                unsafe_allow_html=True)
with h2:
    months = config.target_months()
    month = st.selectbox("Target month (model estimate)", months, index=months.index(config.DEFAULT_MONTH))
with h3:
    st.markdown('<br><span class="pill pill-model">MODEL OUTPUT - not observed data</span>', unsafe_allow_html=True)

w = config.month_warning(month)
if w:
    (st.error if w[0] == "high" else st.warning if w[0] == "elevated" else st.info)(w[1])

fg = config.fullgrid_dir(month)
lst_path, uhi_path = fg / config.FULLGRID_FILES["lst"], fg / config.FULLGRID_FILES["uhi"]
if not (lst_path.exists() and uhi_path.exists()):
    st.error(f"No pre-computed forecast for {month}. Missing: `{fg}`")
    st.code(f"cd C:\\uhi_site\npython scripts\\precompute_months.py {month}", language="powershell")
    st.stop()

st.markdown("##### Explore Bengaluru's urban heat")
left, mid, right = st.columns([1.05, 2.3, 1.15], gap="medium")

# ---------------------------------------------------------------- left: location + reading
with left:
    st.markdown('<div class="eyebrow">01 · Select location</div>', unsafe_allow_html=True)
    q = st.text_input("Search a place", placeholder="MG Road, Bengaluru", label_visibility="collapsed")
    if st.button("Find place") and q.strip():
        try:
            hit = geocode.search(q.strip())
            if hit:
                ss.lat, ss.lon = hit[0], hit[1]
                ss.map_center = [hit[0], hit[1]]
                st.caption(hit[2])
            else:
                st.warning("Nothing found inside Bengaluru for that search.")
        except Exception as e:  # network / rate limit
            st.error(f"Search failed: {e}")
    c1, c2 = st.columns(2)
    ss.lat = c1.number_input("Latitude", value=float(ss.lat), format="%.5f", step=0.001)
    ss.lon = c2.number_input("Longitude", value=float(ss.lon), format="%.5f", step=0.001)
    st.caption("Tip: in Pick point mode, click the map to move the point. In Draw area mode, the reading shows the average of the drawn area.")

    st.markdown('<div class="eyebrow">02 · Current reading</div>', unsafe_allow_html=True)
    poly = ss.drawn_polygon if (ss.drawn_polygon and be.polygon_vertices_in_grid(ss.drawn_polygon)) else None
    have_point = be.in_study_grid(ss.lat, ss.lon)  # used for the marker / preview square on the map
    if poly:
        # An area is drawn: the reading describes the AREA (all valid pixels inside it), not the point.
        lv = raster_io.polygon_values(lst_path, poly)
        uv = raster_io.polygon_values(uhi_path, poly)
        if lv.size == 0 or uv.size == 0:
            st.warning("The drawn area has no valid model pixels (masked or outside the evaluated domain).")
        else:
            label, css = category(float(uv.mean()))
            share1, share2 = float((uv >= 1).mean() * 100), float((uv >= 2).mean() * 100)
            st.markdown(f'<div class="card"><div class="eyebrow">Drawn area · average</div>'
                        f'<div class="bigtemp">{lv.mean():.1f}°C</div>'
                        f'<div class="eyebrow">Predicted surface temp · mean of area</div><br>'
                        f'<span class="pill {css}">{label}</span> &nbsp; mean UHI {uv.mean():+.1f}°C<br><br>'
                        f'<span class="eyebrow">{share1:.0f}% of pixels ≥ +1°C · {share2:.0f}% ≥ +2°C</span></div>',
                        unsafe_allow_html=True)
            st.caption(f"Area reading: {uv.size:,} valid 30 m pixels ({uv.size * 900 / 1e6:.2f} km²). Pixel UHI "
                       f"ranges {uv.min():+.1f} to {uv.max():+.1f}°C. The point coordinates above are not used "
                       "while an area is drawn (press 'Clear drawn area' to go back to the point reading). "
                       "UHI = predicted LST minus the mean predicted LST of the fixed rural mask; labels are "
                       "display bands, not model outputs; model UHI error is about 1°C.")
    elif not have_point:
        s, w_, n, e = be.grid_bounds_wgs84()
        st.error(f"Outside the supported area (lat {s:.2f}-{n:.2f}, lon {w_:.2f}-{e:.2f}).")
    else:
        lst_s = raster_io.sample_point(lst_path, ss.lat, ss.lon)
        uhi_s = raster_io.sample_point(uhi_path, ss.lat, ss.lon)
        if not (lst_s["inside"] and np.isfinite(lst_s["mean"])):
            st.warning("This pixel has no valid model value (masked / outside the evaluated domain).")
        else:
            label, css = category(uhi_s["mean"])
            st.markdown(f'<div class="card"><div class="eyebrow">Selected point</div>'
                        f'<div class="bigtemp">{lst_s["mean"]:.1f}°C</div>'
                        f'<div class="eyebrow">Predicted surface temp · ~150 m mean</div><br>'
                        f'<span class="pill {css}">{label}</span> &nbsp; UHI {uhi_s["mean"]:+.1f}°C</div>',
                        unsafe_allow_html=True)
            st.caption("UHI = predicted LST minus the mean predicted LST of the fixed rural mask. "
                       "Labels are display bands, not model outputs; model UHI error is about 1°C. "
                       "Draw an area on the map to get an area average instead.")

# ---------------------------------------------------------------- right: mitigation controls
with right:
    st.markdown('<div class="eyebrow">03 · Design a cooler block</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Mitigation strategy", list(be.STRATEGIES), format_func=be.STRATEGIES.get)
    half = st.slider("Area half-side (m) - used if you don't draw a polygon", 100, 1500, 500, 50)
    coverage = st.slider("Treatment coverage (% of eligible pixels)", 5, 100, 40, 5)
    kw: dict = {}
    if strategy in ("vegetation_restoration", "bare_soil_restoration"):
        kw["target_vf"] = st.slider("Target vegetation fraction", 0.10, 1.00, 0.60, 0.05)
    if strategy == "reflective_surface":
        kw["target_albedo"] = st.slider("Target albedo", 0.10, 0.60, 0.30, 0.05)
    with st.expander("Advanced (sent only if ticked)"):
        if st.checkbox("Set area cap (m²)"):
            kw["max_area_m2"] = float(st.number_input("max_area_m2", 1000, 5_000_000, 1_000_000, 1000))
        if strategy == "vegetation_restoration" and st.checkbox("Set soil weight"):
            kw["soil_weight"] = st.slider("soil_weight", 0.0, 1.0, 0.5, 0.05)
        if strategy == "reflective_surface" and st.checkbox("Set min impervious fraction"):
            kw["min_impervious_fraction"] = st.slider("min_impervious_fraction", 0.0, 1.0, 0.5, 0.05)
        if strategy == "bare_soil_restoration":
            if st.checkbox("Set min soil fraction"):
                kw["min_soil_fraction"] = st.slider("min_soil_fraction", 0.0, 1.0, 0.5, 0.05)
            if st.checkbox("Set max vegetation fraction"):
                kw["max_veg_fraction"] = st.slider("max_veg_fraction", 0.0, 1.0, 0.3, 0.05)
    run = st.button("Recalculate heat impact", type="primary", use_container_width=True)
    st.caption("Runs the frozen ConvLSTM twice (baseline and modified inputs). Results are model-based "
               "counterfactual estimates, not measured or guaranteed cooling.")

# ---------------------------------------------------------------- centre: map
with mid:
    layer_opts = ["Forecast LST", "Forecast UHI"]
    res: be.ScenarioResult | None = ss.result if (ss.result and ss.result_meta and ss.result_meta["month"] == month) else None
    if res:
        layer_opts += ["Scenario LST", "ΔLST (scenario − baseline)"]
    layer = st.radio("Layer", layer_opts, horizontal=True, label_visibility="collapsed")
    mode = st.radio("Map mode", ["Pick point", "Draw area"], horizontal=True, key="map_mode",
                    help="Pick point: clicks move the point. Draw area: clicks only draw the shape.")

    if layer == "Forecast LST":
        path, sym, cmap = lst_path, False, "RdYlGn_r"
    elif layer == "Forecast UHI":
        path, sym, cmap = uhi_path, True, "RdYlBu_r"
    elif layer == "Scenario LST":
        path, sym, cmap = res.path("scenario"), False, "RdYlGn_r"
    else:
        path, sym, cmap = res.path("delta"), True, "RdBu_r"

    if layer == "Scenario LST":  # same colour range as the scenario's own baseline for a fair comparison
        vmin, vmax = cached_range(str(res.path("baseline")), mt(res.path("baseline")), False)
    else:
        vmin, vmax = cached_range(str(path), mt(path), sym)

    # ---- base map: identical between reruns, otherwise the browser rebuilds it (= flicker)
    m = folium.Map(location=ss.map_center, zoom_start=12, tiles=None, control_scale=True)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Light map", max_zoom=16).add_to(m)
    folium.TileLayer(tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                     attr="&copy; OpenStreetMap contributors", name="Street (OSM)", show=False).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri", name="Satellite", show=False).add_to(m)
    rgba, bounds = cached_overlay(str(path), mt(path), vmin, vmax, cmap)
    folium.raster_layers.ImageOverlay(rgba, bounds=bounds, opacity=0.8, name=layer).add_to(m)
    cm = LinearColormap(["#1a9850", "#fee08b", "#d73027"] if cmap == "RdYlGn_r" else
                        ["#2166ac", "#f7f7f7", "#b2182b"] if "Bu" in cmap and sym else
                        ["#2c7bb6", "#ffffbf", "#d7191c"], vmin=vmin, vmax=vmax,
                        caption=("ΔT (°C), centred on 0" if sym else "°C"))
    cm.add_to(m)
    if mode == "Draw area":  # drawing tools only exist in draw mode, so draw clicks never move the point
        Draw(export=False, draw_options={"polygon": True, "rectangle": True, "polyline": False, "circle": False,
                                         "marker": False, "circlemarker": False},
             edit_options={"edit": False, "remove": False}).add_to(m)
    folium.LayerControl().add_to(m)

    # ---- things that change (point, chosen area) go in a FeatureGroup, so changing them does NOT rebuild the map
    fg = folium.FeatureGroup(name="Selection")
    if have_point:
        folium.CircleMarker([ss.lat, ss.lon], radius=7, color="#14352F", fill=True, fill_color="#F26A45").add_to(fg)
    if ss.drawn_polygon:
        folium.GeoJson(ss.drawn_polygon, style_function=lambda _: {
            "color": "#F26A45", "weight": 3, "fillOpacity": 0.10}).add_to(fg)
    elif have_point:
        folium.GeoJson(be.square_polygon_geojson(ss.lat, ss.lon, half), style_function=lambda _: {
            "color": "#1D5C52", "weight": 2, "dashArray": "6", "fillOpacity": 0.05}).add_to(fg)

    if ss.drawn_polygon:
        minx, miny, maxx, maxy = shape(ss.drawn_polygon).bounds
        m.fit_bounds([[miny, minx], [maxy, maxx]], padding=(30, 30), max_zoom=15)

    out = st_folium(m, key=f"map_{month}_{layer}_{mode}_{ss.map_nonce}", feature_group_to_add=fg, height=520,
                    use_container_width=True, returned_objects=["last_clicked", "all_drawings"])

    if out:
        if mode == "Pick point" and out.get("last_clicked"):
            lc = out["last_clicked"]
            click = (round(lc["lat"], 6), round(lc["lng"], 6))
            if click != ss.last_click:  # act on each click once; a stale click must not override typed coordinates
                ss.last_click = click
                ss.lat, ss.lon = click
                st.rerun()
        if mode == "Draw area" and out.get("all_drawings"):
            g = out["all_drawings"][-1].get("geometry")
            if g and g.get("type") == "Polygon" and g != ss.drawn_polygon:
                ss.drawn_polygon = g      # only ONE area is kept: the newest one replaces the old one
                ss.map_nonce += 1         # rebuild the map so the browser's drawing layer is empty again
                st.rerun()

    if ss.drawn_polygon:
        if be.polygon_vertices_in_grid(ss.drawn_polygon):
            st.success(f"Drawn area: {polygon_area_km2(ss.drawn_polygon):.2f} km² (orange outline). "
                       "It will be used for the simulation.")
        else:
            st.error("The drawn area extends outside the model grid. Clear it and draw inside the coloured area.")
        if st.button("Clear drawn area"):
            ss.drawn_polygon = None
            ss.map_nonce += 1             # rebuild: removes the old shapes still held by the drawing layer
            st.rerun()
    else:
        st.caption("No area drawn: the simulation uses the dashed square around the selected point "
                   "(size set at right). Switch to 'Draw area' to draw your own.")

# ---------------------------------------------------------------- run scenario
if run:
    drawn = ss.drawn_polygon
    polygon = drawn or be.square_polygon_geojson(ss.lat, ss.lon, half)
    try:
        with st.spinner("Running the frozen model (baseline + scenario) - this can take a minute..."):
            ss.result = be.run_scenario(strategy, polygon, month, float(coverage), **kw)
        ss.result_meta = {"month": month, "strategy": strategy, "polygon": polygon, "coverage": coverage,
                          "params": kw, "drawn": bool(drawn)}
        st.rerun()
    except Exception as e:
        st.error(f"Simulation failed: {e}")

# ---------------------------------------------------------------- bottom: before vs after
st.divider()
st.markdown('<div class="eyebrow">Projected outcome</div>', unsafe_allow_html=True)
if not (ss.result and ss.result_meta and ss.result_meta["month"] == month):
    st.info("Pick a strategy and press **Recalculate heat impact** to see baseline vs scenario for the chosen month.")
else:
    res, meta = ss.result, ss.result_meta
    poly = meta["polygon"]
    base_p, scen_p, del_p = res.path("baseline"), res.path("scenario"), res.path("delta")
    b_stats, s_stats, d_stats = (raster_io.polygon_stats(p, poly) for p in (base_p, scen_p, del_p))
    vmin, vmax = cached_range(str(base_p), mt(base_p), False)
    dmin, dmax = cached_range(str(del_p), mt(del_p), True)

    st.subheader("See the difference your plan could make")
    st.caption(f"{be.STRATEGIES[meta['strategy']]} · {month} · coverage {meta['coverage']}% · "
               f"{'drawn polygon' if meta['drawn'] else 'square around the selected point'}")
    c1, c2, c3 = st.columns(3)
    for col, title, p, vlo, vhi, cm_ in ((c1, "Baseline (model)", base_p, vmin, vmax, "RdYlGn_r"),
                                         (c2, "With mitigation (model)", scen_p, vmin, vmax, "RdYlGn_r"),
                                         (c3, "ΔLST = scenario − baseline", del_p, dmin, dmax, "RdBu_r")):
        img, _ = cached_overlay(str(p), mt(p), vlo, vhi, cm_)
        col.image(img, caption=f"{title} · {vlo:+.1f} to {vhi:+.1f} °C" if cm_ == "RdBu_r"
                  else f"{title} · {vlo:.1f} to {vhi:.1f} °C", use_container_width=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Baseline mean LST (inside polygon)", f"{b_stats['mean']:.2f} °C")
    k2.metric("Scenario mean LST (inside polygon)", f"{s_stats['mean']:.2f} °C")
    k3.metric("Mean ΔLST (inside polygon)", f"{d_stats['mean']:+.3f} °C")
    k4.metric("Valid area counted", f"{d_stats['area_m2']:,.0f} m²")
    st.caption(f"ΔLST inside the polygon ranges {d_stats['min']:+.2f} to {d_stats['max']:+.2f} °C at individual "
               "pixels - a single-pixel extreme is not the effect of the whole intervention. "
               "Scope: pixels inside the polygon with valid model output; invalid pixels are excluded, not zero-filled. "
               "ΔUHI is not shown here: the scenario CLI writes LST rasters only.")
    with st.expander("Raw model report (scenario_report.json)"):
        st.json(res.report)
    d1, d2 = st.columns(2)
    d1.download_button("Download scenario_report.json", json.dumps(res.report, indent=2), "scenario_report.json")
    d2.download_button("Download ΔLST GeoTIFF", del_p.read_bytes(), "delta_lst_celsius.tif")
    if st.button("Reset scenario"):
        ss.result = ss.result_meta = None
        st.rerun()