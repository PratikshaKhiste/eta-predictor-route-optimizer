# 🚴 Delivery ETA Predictor and Route Optimizer

An end-to-end AI/ML delivery optimization system that:

1. **Predicts Leg Delivery Time (ETA)** using a trained **XGBoost** regression model incorporating distance, weather, traffic, rider profile, time of day, and location clusters.
2. **Builds an Asymmetric ETA Cost Matrix** between a depot and multiple delivery destinations.
3. **Finds an Optimal Delivery Sequence** using exact Dynamic Programming (**Held-Karp**) or fast local search (**Hill Climbing**).
4. **Exposes the Pipeline via FastAPI** with complete schema validation, baseline route benchmarking, and distance-range guardrails.
5. **Provides an Interactive Streamlit UI** featuring side-by-side route maps (Folium), turn-by-turn arrival schedules, local optima simulations, and model performance comparisons.

---

## 1. Project Architecture

```text
       ┌────────────────────────────────────────────────────────┐
       │                Streamlit Frontend (UI)                 │
       │    - Route Optimizer (Folium Maps & Schedule)          │
       │    - Local Optima Multi-Start Demo                     │
       │    - Model Benchmark & Feature Importance              │
       └──────────────────────────┬─────────────────────────────┘
                                  │ HTTP POST /optimize
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │                 FastAPI Backend Service                │
       │              (api/main.py, api/schemas.py)             │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │                     ETA Matrix Builder                 │
       │   - Pairwise distance calculation (Haversine km)       │
       │   - Feature engineering (weather, traffic, rider, time)│
       │   - XGBoost model inference (models/predict.py)        │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │                    Route Optimizer                     │
       │     ┌─────────────────────┬──────────────────────┐     │
       │     │   Held-Karp (Exact) │ Hill Climbing (Fast) │     │
       │     │   O(n^2 * 2^n) DP   │   2-opt Local Search │     │
       │     └─────────────────────┴──────────────────────┘     │
       └──────────────────────────┬─────────────────────────────┘
                                  │
                                  ▼
       ┌────────────────────────────────────────────────────────┐
       │   Optimized Sequence + Baseline Delta + Arrival ETAs   │
       └────────────────────────────────────────────────────────┘
```

---

## 2. Requirements & Prerequisites

- **Python**: **3.12** recommended (Python 3.13 may have package compatibility issues with certain binary wheels).
- **Git**
- **Operating System**: Windows, macOS, or Linux (PowerShell commands documented below).

---

## 3. Clone and Setup

Open PowerShell or terminal in your projects directory:

```powershell
git clone https://github.com/ShreyaTheExplorer/2_eta_predictor-route_optimizer.git
cd 2_eta_predictor-route_optimizer
```

*(Or if working from the original upstream repository: `https://github.com/Harshada-pande/eta_predictor-route_optimizer.git`)*

---

## 4. Virtual Environment Setup

Create and activate a virtual environment:

```powershell
# Create virtual environment
python -m venv .venv

# Activate in PowerShell
.\.venv\Scripts\Activate.ps1
```

> **Note for PowerShell execution policy:**
> If PowerShell blocks script execution, run:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
> .\.venv\Scripts\Activate.ps1
> ```

---

## 5. Install Dependencies

Install all core ML, optimization, backend, and frontend packages:

```powershell
# Core ML, preprocessing, and test dependencies
pip install -r requirements.txt

# FastAPI, Uvicorn, Streamlit, and UI visualization dependencies
pip install -r requirements-app.txt
```

---

## 6. Verify Installation & Tests

### Automated Unit Test Suite
Run the full test suite covering data cleaning, feature parity, model predictions, optimization algorithms, API routes, and UI helpers:

```powershell
python -m pytest -q
```

**Expected Output:**
```text
........................................................................ [ 71%]
.............................                                            [100%]
101 passed, 1 warning in ~14s
```
*(The warning is a minor Starlette/httpx deprecation notice that does not affect execution.)*

### Live Integration Verification
With the backend running, verify end-to-end integration:

```powershell
python tests/verify_integration.py
```

---

## 7. Running the Application

To run the complete system, launch both the backend API and frontend UI in separate terminals:

### Terminal 1: FastAPI Backend
```powershell
cd path\to\eta_predictor-route_optimizer
.\.venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
- API Base URL: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

### Terminal 2: Streamlit Frontend
```powershell
cd path\to\eta_predictor-route_optimizer
.\.venv\Scripts\Activate.ps1
streamlit run frontend/app.py --server.port 8501
```
- Web Application URL: `http://127.0.0.1:8501`

---

## 8. Streamlit Frontend Features

The Streamlit UI provides three comprehensive tabs:

### 1. 🗺️ Route Optimizer
- **Depot & Stops Configuration**: Quick-select from 15 pre-configured Pune landmarks (e.g., Koregaon Park, Shivajinagar, Kothrud, Aundh, Hinjewadi) or define custom coordinates.
- **Rider & Environmental Controls**: Configure vehicle type (Motorcycle, Scooter, Bicycle, Electric Scooter), vehicle condition, rider age & rating, weather conditions (Sunny, Cloudy, Fog, Stormy), traffic density (Low, Medium, High, Jam), festival indicator, and departure time.
- **Algorithm Selector**: Switch between Held-Karp (exact global minimum) and Hill Climbing (fast heuristic).
- **Side-by-Side Folium Maps**: Visual comparison of the unoptimized baseline sequence vs the optimized delivery route with numbered markers and depot indicators.
- **Key Metrics**: Baseline duration, optimized duration, time saved, and percentage efficiency improvement.
- **Per-Stop Schedule Table**: Detailed breakdown showing leg duration, cumulative ETA, and estimated arrival clock time for each stop.
- **Out-of-Range Warnings**: Alerts if stop distances exceed the training bounds of the model (1.47 km – 20.97 km).

### 2. ⛰️ Local Optima Demo
- Interactive multi-start simulation illustrating why heuristic optimization can get trapped in sub-optimal local minima.
- Generates a random-start cost distribution plot contrasting best vs worst discovered routes against the global optimum.

### 3. 📊 Model Comparison
- Comprehensive benchmark reports comparing Linear Baseline, Random Forest (depths 10, 15, 20), and XGBoost (depths 6, 8, 10).
- Displays evaluation metrics (MAE, RMSE, R²) and feature importance breakdowns.

---

## 9. API Reference

### Health Check
```http
GET /health
```
**Response:**
```json
{
  "status": "ok"
}
```

### Route Optimization
```http
POST /optimize
```

#### Request Payload
```json
{
  "depot_lat": 18.5204,
  "depot_lon": 73.8567,
  "rider": {
    "vehicle_type": "Motorcycle",
    "vehicle_condition": 2,
    "delivery_person_age": 25.0,
    "delivery_person_rating": 4.5
  },
  "festival": "No",
  "city_type": "Metropolitan",
  "stops": [
    {
      "id": "Koregaon Park",
      "lat": 18.5362,
      "lon": 73.8938,
      "weather": "Sunny",
      "traffic": "Medium",
      "multiple_deliveries": 0
    },
    {
      "id": "Shivajinagar",
      "lat": 18.5308,
      "lon": 73.8474,
      "weather": "Sunny",
      "traffic": "Medium",
      "multiple_deliveries": 0
    },
    {
      "id": "Kothrud",
      "lat": 18.5074,
      "lon": 73.8077,
      "weather": "Sunny",
      "traffic": "Medium",
      "multiple_deliveries": 0
    }
  ],
  "departure_time": "2026-10-26T11:00:00",
  "algorithm": "held_karp"
}
```

#### Response Payload
```json
{
  "algorithm": "held_karp",
  "route": [
    "depot",
    "Shivajinagar",
    "Kothrud",
    "Koregaon Park"
  ],
  "total_minutes": 45.71,
  "baseline_route": [
    "depot",
    "Koregaon Park",
    "Shivajinagar",
    "Kothrud"
  ],
  "baseline_total_minutes": 46.20,
  "improvement_pct": 1.06,
  "stops": [
    {
      "id": "Shivajinagar",
      "lat": 18.5308,
      "lon": 73.8474,
      "leg_minutes": 15.14,
      "eta_minutes": 15.14,
      "arrival_time": "2026-10-26T11:15:08"
    },
    {
      "id": "Kothrud",
      "lat": 18.5074,
      "lon": 73.8077,
      "leg_minutes": 15.23,
      "eta_minutes": 30.37,
      "arrival_time": "2026-10-26T11:30:22"
    },
    {
      "id": "Koregaon Park",
      "lat": 18.5362,
      "lon": 73.8938,
      "leg_minutes": 15.34,
      "eta_minutes": 45.71,
      "arrival_time": "2026-10-26T11:45:42"
    }
  ],
  "distance_warnings": {
    "count": 0,
    "training_range_km": [1.47, 20.97],
    "legs": []
  }
}
```

---

## 10. Optimization Algorithms

| Algorithm | Type | Time Complexity | Best Suited For | Guarantee |
| :--- | :--- | :--- | :--- | :--- |
| **Held-Karp** | Dynamic Programming | $\mathcal{O}(n^2 \cdot 2^n)$ | Small-to-medium routes ($\le 12$ stops) | **Globally Optimal** |
| **Hill Climbing** | 2-opt Local Search Heuristic | $\mathcal{O}(k \cdot n^2)$ | Large stops ($\ge 13$ stops), instant results | Fast, but may hit local optima |

---

## 11. Model Evaluation & Benchmark Results

### XGBoost ETA Predictor (Held-Out Test Set)

| Metric | Result | Description |
| :--- | :---: | :--- |
| **MAE** | **3.09 min** | Mean Absolute Error |
| **RMSE** | **3.87 min** | Root Mean Squared Error |
| **$R^2$ Score** | **0.8271** | Explains **82.71%** of variance in delivery time |
| **Classification Accuracy ($\pm 5$ min delay)** | **93.17%** | Reliability on on-time delivery detection |
| **Classification F1 Score** | **87.14%** | Harmonic mean of delay precision and recall |

### Route Optimization Benchmark (100 synthetic cases, 10 delivery stops)

| Metric | Held-Karp (Exact) | Hill Climbing (Heuristic) |
| :--- | :---: | :---: |
| **Optimality Guarantee** | 100% Global Optimum | Suboptimal in 94/100 cases |
| **Mean Optimality Gap** | 0.00% | 0.77% |
| **Maximum Observed Gap** | 0.00% | 5.13% |
| **Mean Runtime** | 26.83 ms | 3.62 ms |

---

## 12. Project Directory Structure

```text
eta_predictor-route_optimizer/
├── api/
│   ├── __init__.py
│   ├── main.py              # FastAPI app endpoints (/health, /optimize)
│   ├── matrix_builder.py    # Pairwise ETA cost matrix constructor
│   └── schemas.py           # Pydantic models & validation rules
├── configs/
│   └── config.yaml          # Project configuration & dataset paths
├── data/
│   ├── raw/                 # Raw dataset files
│   └── processed/           # Feature matrices (train/test)
├── docs/
│   ├── benchmark_results.md # Route optimization benchmarks
│   ├── cleaning_report.md   # Data cleaning documentation
│   └── model_comparison.md  # Model evaluation and comparison
├── evaluation/
│   └── metrics.py           # MAE, RMSE, R², and classification metrics
├── frontend/
│   ├── __init__.py
│   └── app.py               # Streamlit application with Folium maps
├── models/
│   ├── artifacts/
│   │   ├── best_eta_model.joblib   # Trained XGBoost model artifact
│   │   └── feature_columns.json    # Exact model feature columns
│   ├── predict.py           # Single & batch inference routines
│   └── train.py             # Model training & experiment tracking pipeline
├── optimizer/
│   ├── base.py              # Base optimizer interface
│   ├── held_karp.py         # Held-Karp exact dynamic programming
│   ├── hill_climbing.py     # Hill Climbing 2-opt local search
│   ├── local_optima_demo.py # Simulation generator for local optima demo
│   └── registry.py          # Algorithm factory and registry
├── preprocessing/
│   ├── cleaning.py          # Raw food-delivery dataset cleaning rules
│   ├── feature_builder.py   # Transformation from API input to model features
│   ├── prepare_data.py      # Dataset preparation script
│   └── weather.py           # Weather and seasonal feature mappings
├── tests/
│   ├── test_api.py          # FastAPI route and schema tests
│   ├── test_clean.py        # Data cleaning unit tests
│   ├── test_features.py     # Feature engineering tests
│   ├── test_frontend.py     # Streamlit helpers and UI tests
│   ├── test_held_karp.py    # Held-Karp correctness tests
│   ├── test_optimizer.py    # Optimizer benchmark tests
│   ├── test_predict.py      # Inference verification tests
│   └── verify_integration.py# Live end-to-end integration test
├── requirements.txt         # Core ML & backend dependencies
├── requirements-app.txt     # Web application & frontend dependencies
└── README.md
```

---

## 13. Git Workflow & Pushing to GitHub (PowerShell)

To push your latest changes and commits to GitHub:

```powershell
cd c:\Users\VICTUS\OneDrive\Desktop\clg_proj\2_eta\eta_predictor-route_optimizer

# 1. Check current status
git status

# 2. Stage updated files
git add README.md tests/verify_integration.py

# 3. Commit changes
git commit -m "docs: update README with complete architecture, features, and setup instructions"

# 4. Push to remote repository
git push origin master
```

---

## 14. Project Status

### Backend
- [x] Data cleaning and preparation pipeline
- [x] Feature engineering with weather & traffic enrichment
- [x] XGBoost regression model training ($R^2 = 0.8271$, $\text{MAE} = 3.09\text{ min}$)
- [x] ETA cost matrix builder
- [x] Held-Karp exact DP algorithm
- [x] Hill Climbing 2-opt heuristic algorithm
- [x] FastAPI REST API with input validation & error handling
- [x] Comprehensive test suite (101 unit tests passing)

### Frontend
- [x] Streamlit multi-tab user interface
- [x] Interactive Folium maps (Baseline vs Optimized route comparison)
- [x] Full REST API integration with FastAPI `/optimize`
- [x] Per-stop arrival schedule table with formatted timestamps
- [x] Distance extrapolation warnings
- [x] Multi-start local optima simulation tab
- [x] Model comparison and feature importance tab
- [x] End-to-end integration verified
