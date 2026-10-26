"""Streamlit UI for Delivery ETA Predictor and Route Optimizer."""

from datetime import date, time, datetime
import os
from pathlib import Path
import random

import folium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from streamlit_folium import st_folium

from optimizer.local_optima_demo import run_local_optima_demo

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"

# Indian city landmarks matching the training domain
LANDMARKS = [
    {"id": "Koregaon Park",       "lat": 18.5362, "lon": 73.8938},
    {"id": "Shivajinagar",        "lat": 18.5308, "lon": 73.8474},
    {"id": "Kothrud",             "lat": 18.5074, "lon": 73.8077},
    {"id": "Aundh",               "lat": 18.5642, "lon": 73.7769},
    {"id": "Baner",               "lat": 18.5590, "lon": 73.7868},
    {"id": "Viman Nagar",         "lat": 18.5679, "lon": 73.9143},
    {"id": "Magarpatta",          "lat": 18.5089, "lon": 73.9260},
    {"id": "Hadapsar",            "lat": 18.5018, "lon": 73.9260},
    {"id": "Pimpri",              "lat": 18.6208, "lon": 73.7986},
    {"id": "Chinchwad",           "lat": 18.6279, "lon": 73.7956},
    {"id": "Wakad",               "lat": 18.5975, "lon": 73.7617},
    {"id": "Hinjewadi",           "lat": 18.5912, "lon": 73.7389},
    {"id": "Sinhagad Road",       "lat": 18.4667, "lon": 73.8167},
    {"id": "Katraj",              "lat": 18.4529, "lon": 73.8616},
    {"id": "Pune Station",        "lat": 18.5286, "lon": 73.8741},
]

# Schema-valid options (matching api/schemas.py)
WEATHER_OPTIONS   = ["Sunny", "Cloudy", "Fog", "Sandstorms", "Stormy", "Windy", "Unknown"]
TRAFFIC_OPTIONS   = ["Low", "Medium", "High", "Jam", "Unknown"]
VEHICLE_OPTIONS   = ["Motorcycle", "Scooter", "Bicycle", "Electric Scooter"]
FESTIVAL_OPTIONS  = ["No", "Yes", "Unknown"]
CITY_OPTIONS      = ["Metropolitan", "Urban", "Semi-Urban"]
ALGORITHM_OPTIONS = ["held_karp", "hill_climbing", "exact"]
ALGORITHM_LABELS  = {
    "held_karp":    "Held-Karp (Exact DP)",
    "hill_climbing": "Hill Climbing (Heuristic)",
    "exact":        "Exact (alias for Held-Karp)",
}

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Delivery ETA & Route Optimizer",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚴 Delivery ETA Predictor & Route Optimizer")
st.caption(
    "XGBoost-based ETA prediction · Held-Karp exact solver · Hill Climbing heuristic"
)

# ---------------------------------------------------------------------------
# Sidebar — Depot & Departure
# ---------------------------------------------------------------------------

st.sidebar.header("📍 Depot & Departure")
depot_lat = st.sidebar.number_input("Depot Latitude",  value=18.5204, format="%.4f", step=0.0001)
depot_lon = st.sidebar.number_input("Depot Longitude", value=73.8567, format="%.4f", step=0.0001)
dep_date  = st.sidebar.date_input("Departure Date",  value=date(2026, 10, 24))
dep_time  = st.sidebar.time_input("Departure Time",  value=time(11, 0))
departure_iso = f"{dep_date.isoformat()}T{dep_time.strftime('%H:%M:%S')}"

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar — Rider Details
# ---------------------------------------------------------------------------

st.sidebar.header("🧑 Rider Details")
vehicle_type      = st.sidebar.selectbox("Vehicle Type",      VEHICLE_OPTIONS, index=0)
vehicle_condition = st.sidebar.selectbox("Vehicle Condition", [0, 1, 2, 3], index=2,
                                         help="0 = poor … 3 = excellent")
rider_age    = st.sidebar.slider("Rider Age",    min_value=20, max_value=39, value=25)
rider_rating = st.sidebar.slider("Rider Rating", min_value=2.5, max_value=5.0, value=4.5, step=0.1)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar — Route Settings
# ---------------------------------------------------------------------------

st.sidebar.header("⚙️ Route Settings")
festival  = st.sidebar.selectbox("Festival Day",  FESTIVAL_OPTIONS, index=0)
city_type = st.sidebar.selectbox("City Type",     CITY_OPTIONS,     index=0)
algorithm = st.sidebar.selectbox(
    "Optimizer Algorithm",
    ALGORITHM_OPTIONS,
    format_func=lambda k: ALGORITHM_LABELS[k],
    index=0,
)

st.sidebar.divider()

# ---------------------------------------------------------------------------
# Sidebar — Stops Configuration
# ---------------------------------------------------------------------------

st.sidebar.header("🗺️ Stops Configuration")
num_stops = st.sidebar.slider("Number of stops (N)", min_value=1, max_value=12, value=4)
seed      = st.sidebar.number_input("Random Seed", value=42, step=1)


def _sample_stops(n: int, s: int | None = None) -> list[dict]:
    count = min(max(1, int(n)), len(LANDMARKS))
    if s is not None:
        return [dict(r) for r in random.Random(int(s)).sample(LANDMARKS, count)]
    return [dict(r) for r in random.sample(LANDMARKS, count)]


# Per-stop defaults for weather/traffic/multiple_deliveries
def _enrich_stops(stops: list[dict]) -> list[dict]:
    out = []
    for r in stops:
        out.append({
            "id":                 r.get("id", ""),
            "lat":                r.get("lat", 0.0),
            "lon":                r.get("lon", 0.0),
            "weather":            r.get("weather", "Sunny"),
            "traffic":            r.get("traffic", "Medium"),
            "multiple_deliveries": int(r.get("multiple_deliveries", 0)),
        })
    return out


params = (int(num_stops), int(seed))

if "sample_ver" not in st.session_state:
    st.session_state["sample_ver"]   = 0
    st.session_state["stops"]        = _enrich_stops(_sample_stops(num_stops, seed))
    st.session_state["last_params"]  = params
elif st.session_state.get("last_params") != params:
    st.session_state["stops"]       = _enrich_stops(_sample_stops(num_stops, seed))
    st.session_state["sample_ver"] += 1
    st.session_state["last_params"] = params
    st.session_state.pop("optimize_result",        None)
    st.session_state.pop("last_optimized_payload", None)

if st.sidebar.button("🔀 Sample Random Stops"):
    st.session_state["stops"]       = _enrich_stops(_sample_stops(num_stops, None))
    st.session_state["sample_ver"] += 1
    st.session_state.pop("optimize_result",        None)
    st.session_state.pop("last_optimized_payload", None)

# Editable stops table
editor_data = st.sidebar.data_editor(
    st.session_state["stops"],
    num_rows="dynamic",
    column_config={
        "id":                  st.column_config.TextColumn("ID"),
        "lat":                 st.column_config.NumberColumn("Lat", format="%.4f"),
        "lon":                 st.column_config.NumberColumn("Lon", format="%.4f"),
        "weather":             st.column_config.SelectboxColumn("Weather",  options=WEATHER_OPTIONS),
        "traffic":             st.column_config.SelectboxColumn("Traffic",  options=TRAFFIC_OPTIONS),
        "multiple_deliveries": st.column_config.NumberColumn("Multi-Del", min_value=0, max_value=3, step=1),
    },
    key=f"stops_editor_{st.session_state['sample_ver']}",
)

# ---------------------------------------------------------------------------
# Map helper
# ---------------------------------------------------------------------------

def build_map(depot: dict, stops_coords: dict, route_order: list, color: str = "#1976D2") -> folium.Map:
    coords = {"depot": (depot["lat"], depot["lon"]), **stops_coords}
    pts    = [coords[n] for n in route_order if n in coords]
    center = (
        [sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)]
        if pts else [depot["lat"], depot["lon"]]
    )
    m = folium.Map(location=center, zoom_start=13)
    if len(pts) > 1:
        folium.PolyLine(pts, color=color, weight=4, opacity=0.85).add_to(m)

    depot_html = (
        '<div style="font-size:10px;font-weight:bold;color:white;background:#C62828;'
        'border-radius:4px;padding:2px 8px;border:2px solid white;'
        'box-shadow:1px 1px 3px rgba(0,0,0,.5);">DEPOT</div>'
    )
    folium.Marker(
        [depot["lat"], depot["lon"]],
        tooltip="Depot (Start)",
        icon=folium.DivIcon(html=depot_html),
    ).add_to(m)

    for i, node in enumerate(route_order):
        if node == "depot" or node not in coords:
            continue
        badge = (
            f'<div style="font-size:11px;font-weight:bold;color:white;background:{color};'
            f'border-radius:50%;width:24px;height:24px;display:flex;align-items:center;'
            f'justify-content:center;border:2px solid white;'
            f'box-shadow:1px 1px 3px rgba(0,0,0,.5);">{i}</div>'
        )
        folium.Marker(
            coords[node],
            tooltip=f"{i}. {node}",
            icon=folium.DivIcon(html=badge),
        ).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_route, tab_demo, tab_model = st.tabs(
    ["Route Optimizer", "Local Optima Demo", "Model Comparison"]
)

# ===========================================================================
# Tab 1 — Route Optimizer
# ===========================================================================

with tab_route:
    # Build stops list from editor
    raw_stops = (
        editor_data.to_dict(orient="records")
        if isinstance(editor_data, pd.DataFrame)
        else list(editor_data)
    )

    stops_list = []
    for r in raw_stops:
        sid = str(r.get("id", "")).strip()
        if not sid:
            continue
        try:
            stops_list.append({
                "id":                  sid,
                "lat":                 float(r.get("lat", 0)),
                "lon":                 float(r.get("lon", 0)),
                "weather":             str(r.get("weather", "Sunny")),
                "traffic":             str(r.get("traffic", "Medium")),
                "multiple_deliveries": int(r.get("multiple_deliveries", 0)),
            })
        except (ValueError, TypeError):
            continue

    # Validation guards
    if len(stops_list) > 15:
        st.error(
            f"⚠️ Too many stops ({len(stops_list)}). The API supports at most 15."
        )
    elif len(stops_list) > 12:
        st.warning(
            f"⚠️ {len(stops_list)} stops — Held-Karp will still work but may be slow."
        )

    # Duplicate ID check
    ids = [s["id"] for s in stops_list]
    if len(ids) != len(set(ids)):
        st.error("⚠️ Duplicate stop IDs detected. Each stop must have a unique ID.")

    can_optimize = (
        1 <= len(stops_list) <= 15
        and len(ids) == len(set(ids))
    )

    # Full API payload matching api/schemas.py exactly
    current_payload = {
        "depot_lat":      float(depot_lat),
        "depot_lon":      float(depot_lon),
        "rider": {
            "vehicle_type":            vehicle_type,
            "vehicle_condition":       vehicle_condition,
            "delivery_person_age":     float(rider_age),
            "delivery_person_rating":  float(rider_rating),
        },
        "festival":        festival,
        "city_type":       city_type,
        "stops":           stops_list,
        "departure_time":  departure_iso,
        "algorithm":       algorithm,
    }

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        clicked = st.button(
            "🚀 Optimize Route",
            type="primary",
            disabled=not can_optimize,
            use_container_width=True,
        )
    with col_info:
        if not can_optimize:
            st.info("Fix validation issues above before optimizing.")
        else:
            st.success(
                f"Ready — {len(stops_list)} stop(s) · {ALGORITHM_LABELS[algorithm]}"
            )

    if clicked:
        with st.spinner("Running ETA prediction & route optimization…"):
            try:
                resp = requests.post(
                    f"{API_URL}/optimize",
                    json=current_payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    st.session_state["optimize_result"]        = resp.json()
                    st.session_state["last_optimized_payload"] = current_payload
                    st.session_state["last_depot"]             = {"lat": float(depot_lat), "lon": float(depot_lon)}
                elif resp.status_code == 422:
                    detail = resp.json().get("detail", resp.text)
                    errs = (
                        [
                            f"• **{' → '.join(str(x) for x in d.get('loc', []) if x != 'body')}**: "
                            f"{d.get('msg', 'Invalid value')}"
                            for d in detail
                        ]
                        if isinstance(detail, list)
                        else [str(detail)]
                    )
                    st.error("Validation Error (422):\n\n" + "\n".join(errs))
                else:
                    st.error(f"API Error ({resp.status_code}): {resp.text}")
            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Cannot reach the API at **{API_URL}**.\n\n"
                    "Start the backend with:\n"
                    "```\npython -m uvicorn api.main:app --reload\n```"
                )
            except requests.exceptions.Timeout:
                st.error("⏱️ Request timed out (60 s). Try fewer stops or a simpler algorithm.")
            except requests.exceptions.RequestException as exc:
                st.error(f"Request failed: {exc}")

    result      = st.session_state.get("optimize_result")
    last_payload = st.session_state.get("last_optimized_payload")
    last_depot   = st.session_state.get("last_depot", {"lat": float(depot_lat), "lon": float(depot_lon)})

    if result and last_payload and last_payload != current_payload:
        st.warning(
            "⚠️ **Settings changed** since the last run. "
            "Results below reflect the previous configuration. "
            "Click **Optimize Route** to recalculate."
        )

    if result:
        st.divider()

        # ── Summary metrics ────────────────────────────────────────────────
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Algorithm",         ALGORITHM_LABELS.get(result.get("algorithm", ""), result.get("algorithm", "")))
        m2.metric("Baseline Duration", f"{result['baseline_total_minutes']:.1f} min")
        m3.metric("Optimized Duration", f"{result['total_minutes']:.1f} min")

        imp = result.get("improvement_pct", 0)
        delta_str = f"{imp:+.2f}%"
        m4.metric("Improvement", f"{imp:.2f}%", delta=delta_str if imp != 0 else None,
                  delta_color="normal" if imp >= 0 else "inverse")

        # ── Distance warnings ──────────────────────────────────────────────
        dw = result.get("distance_warnings", {})
        if dw.get("count", 0) > 0:
            tr = dw.get("training_range_km", [])
            st.warning(
                f"⚠️ **{dw['count']} leg(s)** had distances outside the training range "
                f"({tr[0]}–{tr[1]} km) and were clipped. Predictions for these legs are extrapolations."
            )

        # ── Route sequences ────────────────────────────────────────────────
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.markdown("**Baseline Route**")
            st.write(" → ".join(result.get("baseline_route", [])))
        with col_r2:
            st.markdown("**Optimized Route**")
            st.write(" → ".join(result.get("route", [])))

        # ── Maps ───────────────────────────────────────────────────────────
        stops_coords = {s["id"]: (s["lat"], s["lon"]) for s in result.get("stops", [])}

        col_before, col_after = st.columns(2)
        with col_before:
            st.subheader("Before: Baseline Route")
            m_before = build_map(last_depot, stops_coords, result["baseline_route"], color="#757575")
            st_folium(
                m_before,
                width=None, height=420,
                returned_objects=[],
                use_container_width=True,
                key="map_before",
            )
        with col_after:
            st.subheader("After: Optimized Route")
            m_after = build_map(last_depot, stops_coords, result["route"], color="#2E7D32")
            st_folium(
                m_after,
                width=None, height=420,
                returned_objects=[],
                use_container_width=True,
                key="map_after",
            )

        # ── Per-stop schedule ──────────────────────────────────────────────
        st.subheader("📋 Per-Stop Delivery Schedule")
        if result.get("stops"):
            schedule_df = pd.DataFrame(result["stops"])
            # Rename columns for display
            rename = {
                "id":           "Stop ID",
                "lat":          "Latitude",
                "lon":          "Longitude",
                "leg_minutes":  "Leg (min)",
                "eta_minutes":  "ETA (min)",
                "arrival_time": "Arrival Time",
            }
            display_cols = [c for c in rename if c in schedule_df.columns]
            schedule_df = schedule_df[display_cols].rename(columns=rename)
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

        # ── Raw JSON (expandable) ──────────────────────────────────────────
        with st.expander("🔍 Full API Response (JSON)"):
            st.json(result)

# ===========================================================================
# Tab 2 — Local Optima Demo
# ===========================================================================

with tab_demo:
    st.caption(
        "Hill climbing from random starting routes. Costs come from a synthetic "
        "asymmetric time matrix (seeded), sized to the current number of stops + depot."
    )

    starts = st.slider("Number of random starts", 10, 100, 50, key="demo_starts")
    demo_n = max(num_stops + 1, 4)
    demo   = run_local_optima_demo(num_starts=starts, n_stops=demo_n, seed=int(seed))

    costs = [r["final_cost"] for r in demo]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(
        range(1, len(costs) + 1),
        costs,
        marker="o",
        linewidth=1,
        markersize=4,
        label="Final hill-climbing cost",
    )
    ax.axhline(min(costs), linestyle="--", color="green",  label=f"Best = {min(costs):.2f} min")
    ax.axhline(max(costs), linestyle="--", color="red",    label=f"Worst = {max(costs):.2f} min")
    ax.set_xlabel("Random starting route")
    ax.set_ylabel("Final route cost (minutes)")
    ax.set_title("Hill Climbing — Effect of Starting Route on Final Cost")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    unique_costs = len(set(round(c, 6) for c in costs))
    st.write(
        f"**{unique_costs}** unique final costs out of {len(costs)} runs → "
        f"hill climbing gets trapped in **{len(costs) - unique_costs}** duplicate local optima."
    )

# ===========================================================================
# Tab 3 — Model Comparison
# ===========================================================================

with tab_model:
    md_path = DOCS_DIR / "model_comparison.md"
    if md_path.exists():
        content = md_path.read_text(encoding="utf-8")
        # Trim internal notes section if present
        if "## Key Findings & Viva Talking Points" in content:
            content = content.split("## Key Findings & Viva Talking Points")[0].strip()
        st.markdown(content)
    else:
        st.warning(
            "📄 `docs/model_comparison.md` not found. "
            "Run `python models/train.py` to generate it."
        )
