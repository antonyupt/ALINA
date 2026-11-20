"""
ConcretoIA — Entrenamiento del modelo de resistencia
Hackathon Innova Sur 2026 — Reto 2 · Cemento Yura S.A.

Dataset: Yeh, I-C. (1998). Concrete Compressive Strength. UCI Machine Learning
Repository, https://doi.org/10.24432/C5PK67 (CC BY 4.0). 1030 ensayos reales.

Uso:
    cd concretoia
    python modelo/train_model.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# ──────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH   = os.path.join(SCRIPT_DIR, "concrete_data.csv")

FEATURES = [
    "cement", "slag", "flyash", "water",
    "superplasticizer", "coarseaggregate", "fineaggregate", "age"
]
TARGET = "csMPa"

# ──────────────────────────────────────────────────────────────────
# Dataset — genera sintético si no existe el CSV
# ──────────────────────────────────────────────────────────────────
def get_dataset() -> pd.DataFrame:
    if os.path.exists(CSV_PATH):
        print(f"OK CSV encontrado: {CSV_PATH}")
        df = pd.read_csv(CSV_PATH)
        # Normalizar nombres de columnas
        df.columns = [c.strip().lower().replace(" ","") for c in df.columns]
        # Mapear si vienen con nombres del UCI
        rename = {
            "cementkg/m3":"cement","blastfurnaceslagkg/m3":"slag",
            "flyashkg/m3":"flyash","waterkg/m3":"water",
            "superplasticizerkg/m3":"superplasticizer",
            "coarseaggregatekg/m3":"coarseaggregate",
            "fineaggregatekg/m3":"fineaggregate",
            "age(day)":"age","concretecompressivestrengthmpamegapascals":"csmpa",
            "concretecompressivestrength":"csmpa",
            # versión pública habitual (concrete.csv)
            "ash":"flyash","superplastic":"superplasticizer",
            "coarseagg":"coarseaggregate","fineagg":"fineaggregate","strength":"csmpa",
        }
        df.rename(columns=rename, inplace=True)
        if "csmpa" in df.columns and TARGET not in df.columns:
            df.rename(columns={"csmpa": TARGET}, inplace=True)
        return _validar_dataset(df)
    raise FileNotFoundError(
        f"No se encontró {CSV_PATH}. Descarga el dataset 'Concrete Compressive Strength' "
        "(UCI, id 165) y guárdalo con columnas: " + ", ".join(FEATURES + [TARGET])
    )


def _validar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rechaza datos que no parezcan ensayos reales y quita réplicas exactas.

    Una versión anterior entrenaba con datos sintéticos recortados a 2 MPa (más de la
    mitad de las filas): el R² era alto, pero el modelo no reflejaba concreto real.
    """
    faltan = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if faltan:
        raise ValueError(f"Faltan columnas en el dataset: {faltan}")
    df = df[FEATURES + [TARGET]].dropna()
    piso = float((df[TARGET] <= df[TARGET].min() + 1e-6).mean())
    if piso > 0.05:
        raise ValueError(
            f"{piso:.0%} de las filas comparten la resistencia mínima ({df[TARGET].min()} MPa): "
            "parece un dataset sintético recortado. Usa ensayos reales."
        )
    duplicadas = int(df.duplicated().sum())
    if duplicadas:
        # Las réplicas exactas caen en train y test a la vez e inflan el R².
        print(f"   Se eliminan {duplicadas} filas duplicadas antes de dividir train/test.")
        df = df.drop_duplicates().reset_index(drop=True)
    return df


# ──────────────────────────────────────────────────────────────────
# Entrenamiento
# ──────────────────────────────────────────────────────────────────
def train():
    print("\n" + "="*55)
    print("  ConcretoIA — Entrenamiento de modelos")
    print("  Cemento Yura S.A. · Innova Sur 2026")
    print("="*55)

    df = get_dataset()
    print(f"\nDataset: {len(df)} muestras, {df.shape[1]} columnas")

    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )

    print("\nEntrenando XGBRegressor (300 estimadores)...")
    model.fit(X_train_s, y_train)

    # Métricas
    y_pred = model.predict(X_test_s)
    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\nMetricas en test:")
    print(f"   R²   = {r2:.4f}")
    print(f"   MAE  = {mae:.3f} MPa")
    print(f"   RMSE = {rmse:.3f} MPa")

    # Cross-validation
    print("\nCross-validation (5-fold)...")
    cv_scores = cross_val_score(
        XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=42,
                     n_jobs=-1, verbosity=0),
        X, y, cv=KFold(5, shuffle=True, random_state=42), scoring='r2', n_jobs=-1
    )
    print(f"   CV R² = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Comparar modelos sobre las mismas particiones y elegir el mejor por CV R².
    candidates = {
        "XGBoost": XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbosity=0,
        ),
        "CatBoost": CatBoostRegressor(
            iterations=300, depth=6, learning_rate=0.05,
            loss_function="RMSE", random_seed=42, verbose=False,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=42, n_jobs=-1,
        ),
    }
    benchmark = {}
    print("\nComparando modelos (5-fold CV)...")
    for name, candidate in candidates.items():
        candidate.fit(X_train_s, y_train)
        candidate_pred = candidate.predict(X_test_s)
        candidate_cv = cross_val_score(
            candidate, X, y, cv=KFold(5, shuffle=True, random_state=42), scoring="r2", n_jobs=1
        )
        benchmark[name] = {
            "r2_test": r2_score(y_test, candidate_pred),
            "mae_mpa": mean_absolute_error(y_test, candidate_pred),
            "rmse_mpa": np.sqrt(mean_squared_error(y_test, candidate_pred)),
            "cv_r2_mean": candidate_cv.mean(),
            "cv_r2_std": candidate_cv.std(),
        }
        print(
            f"   {name:<10} CV R2 = {candidate_cv.mean():.4f} +/- {candidate_cv.std():.4f}"
        )

    selected_name = max(benchmark, key=lambda name: benchmark[name]["cv_r2_mean"])
    model = candidates[selected_name]
    selected_metrics = benchmark[selected_name]
    r2 = selected_metrics["r2_test"]
    mae = selected_metrics["mae_mpa"]
    rmse = selected_metrics["rmse_mpa"]
    candidate_paths = {
        "XGBoost": os.path.join(SCRIPT_DIR, "model_xgboost.pkl"),
        "CatBoost": os.path.join(SCRIPT_DIR, "model_catboost.pkl"),
        "ExtraTrees": os.path.join(SCRIPT_DIR, "model_extratrees.pkl"),
    }
    for name, candidate in candidates.items():
        candidate.fit(X_train_s, y_train)
        joblib.dump(candidate, candidate_paths[name])
    print(f"\nModelo seleccionado: {selected_name}")

    # Importancias
    imp = dict(zip(FEATURES, model.feature_importances_))
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    print("\nImportancia de variables:")
    for name, pct in imp_sorted:
        bar = "#" * max(1, int(pct * 30))
        print(f"   {name:<20} {bar} {pct*100:.1f}%")

    # Guardar
    model_path  = os.path.join(SCRIPT_DIR, "model_xgboost.pkl")
    selected_path = os.path.join(SCRIPT_DIR, "model_selected.pkl")
    scaler_path = os.path.join(SCRIPT_DIR, "scaler.pkl")
    meta_path   = os.path.join(SCRIPT_DIR, "model_meta.json")

    joblib.dump(candidates["XGBoost"], model_path)
    joblib.dump(model, selected_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "modelo": selected_name,
        "n_estimators": 300, "max_depth": 6,
        "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
        "metricas": {
            "r2_test":    round(r2, 4),
            "mae_mpa":    round(mae, 3),
            "rmse_mpa":   round(rmse, 3),
            "cv_r2_mean": round(selected_metrics["cv_r2_mean"], 4),
            "cv_r2_std":  round(selected_metrics["cv_r2_std"], 4),
        },
        "benchmark": {
            name: {key: round(float(value), 4) for key, value in metrics.items()}
            for name, metrics in benchmark.items()
        },
        "features": FEATURES,
        "n_samples_train": len(X_train),
        "n_samples_test":  len(X_test),
        "importancia": {k: round(float(v), 4) for k, v in imp.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nOK Modelo guardado   -> {model_path}")
    print(f"OK Modelo seleccionado -> {selected_path}")
    print(f"OK Scaler guardado   -> {scaler_path}")
    print(f"OK Metadatos         -> {meta_path}")
    print("\nListo. Inicia el backend con:")
    print("   cd concretoia && python -m uvicorn backend.main:app --reload --port 8000\n")

if __name__ == "__main__":
    train()
