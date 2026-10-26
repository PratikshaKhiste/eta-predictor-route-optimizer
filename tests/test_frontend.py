"""Tests for frontend helpers and invariants."""

from frontend.app import LANDMARKS, build_map

# Pune / Indian city bounding box (matches the actual landmarks)
PUNE_LAT_MIN = 18.40
PUNE_LAT_MAX = 18.65
PUNE_LON_MIN = 73.70
PUNE_LON_MAX = 73.95


def test_landmarks_count_and_uniqueness():
    assert len(LANDMARKS) == 15
    ids = [lm["id"] for lm in LANDMARKS]
    assert len(ids) == len(set(ids)), "Landmark IDs must be unique"


def test_landmarks_within_pune_box():
    for lm in LANDMARKS:
        assert PUNE_LAT_MIN <= lm["lat"] <= PUNE_LAT_MAX, f"{lm['id']} lat out of bounds"
        assert PUNE_LON_MIN <= lm["lon"] <= PUNE_LON_MAX, f"{lm['id']} lon out of bounds"


def test_build_map_creates_folium_map():
    depot = {"lat": 18.5204, "lon": 73.8567}
    stops_coords = {
        "s1": (18.5362, 73.8938),
        "s2": (18.5308, 73.8474),
    }
    route = ["depot", "s1", "s2"]
    m = build_map(depot, stops_coords, route)
    assert m is not None
    html = m.get_root().render()
    assert "DEPOT" in html
    assert "s1" in html
    assert "s2" in html


def test_app_test_initial_load():
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app_path = Path(__file__).resolve().parent.parent / "frontend" / "app.py"
    at = AppTest.from_file(str(app_path)).run()
    assert not at.exception
    assert at.title[0].value == "\U0001f6b4 Delivery ETA Predictor & Route Optimizer"
    assert [t.label for t in at.tabs] == [
        "Route Optimizer",
        "Local Optima Demo",
        "Model Comparison",
    ]


