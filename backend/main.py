"""
ConcretoIA — Backend FastAPI
Hackathon Innova Sur 2026 — Reto 2 · Cemento Yura S.A.
"""

import os
import json
import math
import sqlite3
import uuid
import hashlib
import re
from io import BytesIO
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from pypdf import PdfReader

# ──────────────────────────────────────────────────────────────────
# App
# ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ConcretoIA API",
    description="Predicción de resistencia de concreto con XGBoost — Cemento Yura S.A.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv("CONCRETOIA_CORS_ORIGINS", "http://localhost:3000,http://localhost:5500,http://127.0.0.1:5500,null").split(",")
        if origin.strip()
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────────
# Modelo & Scaler
# ──────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "modelo")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")

FEATURES = [
    "cement", "slag", "flyash", "water",
    "superplasticizer", "coarseaggregate", "fineaggregate", "age"
]


def _normalizar_nombre_modelo(nombre: Optional[str]) -> str:
    if nombre is None:
        return "XGBoost"
    nombre = str(nombre).strip()
    return nombre if nombre else "XGBoost"


def _is_scaler_compatible(scaler_obj) -> bool:
    if scaler_obj is None:
        return False
    n_features = getattr(scaler_obj, "n_features_in_", None)
    if n_features is None:
        return True
    return int(n_features) == len(FEATURES)


def _is_model_compatible(model_obj) -> bool:
    if model_obj is None or not hasattr(model_obj, "predict"):
        return False
    n_features = getattr(model_obj, "n_features_in_", None)
    if n_features not in (None, 0):
        return int(n_features) == len(FEATURES)
    feature_names = getattr(model_obj, "feature_names_in_", None)
    if feature_names is not None:
        return list(feature_names) == FEATURES
    return True


def _can_predict_with_expected_features(model_obj) -> bool:
    if model_obj is None or not hasattr(model_obj, "predict"):
        return False
    try:
        sample = np.zeros((1, len(FEATURES)), dtype=float)
        model_obj.predict(sample)
        return True
    except Exception:
        return False


def _load_meta() -> dict:
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "modelo": "XGBoost",
            "n_estimators": 300,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "metricas": {
                "r2_test": 0.864,
                "mae_mpa": 2.832,
                "rmse_mpa": 4.024,
                "cv_r2_mean": 0.8763,
                "cv_r2_std": 0.0072,
            },
            "features": FEATURES,
        }


META = _load_meta()
MODEL_OK = False
model = None
scaler = None
selected_model_name = _normalizar_nombre_modelo(META.get("modelo", "XGBoost"))
AVAILABLE_MODELS = {}
MODEL_FILES = {
    "XGBoost": "model_xgboost.pkl",
    "CatBoost": "model_catboost.pkl",
    "ExtraTrees": "model_extratrees.pkl",
}

for model_name, filename in MODEL_FILES.items():
    path = os.path.join(MODEL_DIR, filename)
    if os.path.exists(path):
        try:
            AVAILABLE_MODELS[model_name] = joblib.load(path)
        except Exception as e:
            print(f"[WARN] No se pudo cargar {model_name}: {e}")

selected_model_path = os.path.join(MODEL_DIR, "model_selected.pkl")
if os.path.exists(selected_model_path):
    try:
        selected_model = joblib.load(selected_model_path)
        AVAILABLE_MODELS.setdefault(selected_model_name, selected_model)
    except Exception as e:
        print(f"[WARN] No se pudo cargar el modelo seleccionado: {e}")

fallback_model_path = os.path.join(MODEL_DIR, "model_xgboost.pkl")
if "XGBoost" in AVAILABLE_MODELS:
    model = AVAILABLE_MODELS["XGBoost"]
else:
    for candidate in AVAILABLE_MODELS.values():
        model = candidate
        break

try:
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
except Exception as e:
    print(f"[WARN] No se pudo cargar el scaler: {e}")

if selected_model_name in AVAILABLE_MODELS:
    model = AVAILABLE_MODELS[selected_model_name]
elif model is not None and _is_model_compatible(model):
    selected_model_name = "XGBoost" if "XGBoost" in AVAILABLE_MODELS else next(iter(AVAILABLE_MODELS), "XGBoost")
    model = AVAILABLE_MODELS.get(selected_model_name, model)

if model is not None and not _can_predict_with_expected_features(model):
    model = None
MODEL_OK = model is not None and scaler is not None and _is_scaler_compatible(scaler) and _can_predict_with_expected_features(model)

if not MODEL_OK:
    print("[WARN] El modelo o el scaler no están disponibles o no son compatibles con las variables del sistema.")


def require_model_ready() -> None:
    if not MODEL_OK or model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Modelo o scaler no disponible, o no son compatibles con las variables esperadas.")
    if not _is_scaler_compatible(scaler):
        raise HTTPException(status_code=503, detail="El scaler no coincide con las variables del modelo.")
    if not _can_predict_with_expected_features(model):
        raise HTTPException(status_code=503, detail="El modelo cargado no puede predecir con las variables esperadas por la API.")


def build_feature_vector(data: dict) -> np.ndarray:
    missing = []
    for field in FEATURES:
        value = data.get(field)
        if value is None:
            missing.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing.append(field)
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"El campo '{field}' debe ser numérico.")
        if not math.isfinite(numeric):
            raise HTTPException(status_code=422, detail=f"El campo '{field}' debe ser un número finito.")
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan datos requeridos para la predicción: {', '.join(missing)}")
    values = [float(data[field]) for field in FEATURES]
    return np.array([values], dtype=float)


def _missing_input_fields(data: dict) -> list:
    missing = []
    for field in FEATURES:
        value = data.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(field)
    return missing


def _breakdown_cost_and_carbon(data: dict) -> dict:
    cost_breakdown = {}
    carbon_breakdown = {}
    for key in FEATURES:
        qty = float(data.get(key, 0) or 0)
        cost_breakdown[key] = round(qty * COST_FACTOR.get(key, 0), 2)
        carbon_breakdown[key] = round(qty * CO2_FACTOR.get(key, 0), 2)
    return {
        "costos_por_material": cost_breakdown,
        "co2_por_material": carbon_breakdown,
        "costo_total_sol_m3": round(sum(cost_breakdown.values()), 2),
        "co2_total_kg_m3": round(sum(carbon_breakdown.values()), 2),
    }

# ──────────────────────────────────────────────────────────────────
# Factores Yura
# ──────────────────────────────────────────────────────────────────
CO2_FACTOR = {
    "cement": 0.820, "slag": 0.050, "flyash": 0.004,
    "water": 0.001,  "superplasticizer": 0.580,
    "coarseaggregate": 0.048, "fineaggregate": 0.045,
}
COST_FACTOR = {          # S/. por kg
    "cement": 0.42,  "slag": 0.18,  "flyash": 0.10,
    "water": 0.001,  "superplasticizer": 2.50,
    "coarseaggregate": 0.08, "fineaggregate": 0.07,
}

EXPOSURE_PROFILES = {
    "normal": {
        "nombre": "Sin exposición especial",
        "ac_max": 0.65,
        "resistencia_min": 210,
        "cemento": "Cemento definido por el diseño estructural",
        "recomendacion": "Mantener control de la relación agua/cementante y curado adecuado.",
    },
    "sulfatos": {
        "nombre": "Sulfatos altos en suelo o agua",
        "ac_max": 0.45,
        "resistencia_min": 280,
        "cemento": "Cemento resistente a sulfatos o sistema aprobado por el responsable estructural",
        "recomendacion": "Confirmar el nivel de sulfatos con un estudio de suelo y especificar el cemento según la severidad de exposición.",
    },
    "cloruros": {
        "nombre": "Cloruros o ambiente marino",
        "ac_max": 0.45,
        "resistencia_min": 280,
        "cemento": "Cemento y sistema de baja permeabilidad definidos por el diseño",
        "recomendacion": "Controlar permeabilidad, recubrimiento, curado y protección contra ingreso de cloruros.",
    },
    "acidos": {
        "nombre": "Medio ácido",
        "ac_max": 0.45,
        "resistencia_min": 280,
        "cemento": "El cemento por sí solo no resuelve el ataque ácido",
        "recomendacion": "Solicitar evaluación especializada y considerar barrera, revestimiento o material resistente al ácido.",
    },
}

def clasificar(kgcm2: float) -> str:
    if kgcm2 < 175:  return "Concreto Simple (< f'c 175)"
    if kgcm2 < 210:  return "Doméstico (f'c 175–210)"
    if kgcm2 < 280:  return "Estructural (f'c 210–280) ✓ NTE E.060"
    if kgcm2 < 350:  return "Alta Resistencia (f'c 280–350)"
    if kgcm2 < 420:  return "Muy Alta Resistencia (f'c 350–420)"
    return "Ultra Alta Resistencia (> f'c 420)"

def calc_extras(m: dict) -> tuple:
    co2  = sum(m.get(k, 0) * CO2_FACTOR.get(k, 0)  for k in CO2_FACTOR)
    cost = sum(m.get(k, 0) * COST_FACTOR.get(k, 0) for k in COST_FACTOR)
    return round(co2, 2), round(cost, 2)

def relacion_agua_cementante(cement: float, slag: float, flyash: float, water: float) -> float:
    """Relación agua/materiales cementantes (a/mc) = agua / (cemento + escoria + ceniza).

    Es la definición usada por NTE E.060 / ACI 318 para los límites de durabilidad.
    Antes se ponderaba la escoria (0.4) y la ceniza (0.3) sin sustento normativo,
    lo que sobrestimaba la a/c de las mezclas con adiciones.
    """
    return float(water) / (float(cement) + float(slag) + float(flyash) + 1e-9)


def _load_training_domain() -> dict:
    """Rango (min, max) de cada variable en el dataset de entrenamiento.

    El modelo solo es confiable dentro de este dominio; fuera de él extrapola.
    """
    path = os.path.join(BASE_DIR, "modelo", "concrete_data.csv")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            header = [c.strip().lower() for c in handle.readline().strip().split(",")]
        data = np.genfromtxt(path, delimiter=",", skip_header=1)
        domain = {}
        for feature in FEATURES:
            col = data[:, header.index(feature)]
            domain[feature] = (float(np.nanmin(col)), float(np.nanmax(col)))
        target = data[:, header.index("csmpa")]
        domain["_resistencia_max_mpa"] = float(np.nanmax(target))
        return domain
    except Exception as exc:  # el dominio es una ayuda: si falla no se bloquea la API
        print(f"[WARN] No se pudo calcular el dominio de entrenamiento: {exc}")
        return {}


TRAINING_DOMAIN = _load_training_domain()


def evaluar_exposicion(exposicion: str, kgcm2: float, ac: float) -> dict:
    perfil = EXPOSURE_PROFILES.get(exposicion, EXPOSURE_PROFILES["normal"])
    alertas = []
    if ac > perfil["ac_max"]:
        alertas.append(f"La relación a/c ({ac:.2f}) supera el valor orientativo de {perfil['ac_max']:.2f} para este ambiente.")
    if kgcm2 < perfil["resistencia_min"]:
        alertas.append(f"La resistencia estimada ({kgcm2:.0f} kg/cm²) está por debajo del umbral orientativo de {perfil['resistencia_min']} kg/cm².")
    return {
        "perfil": exposicion if exposicion in EXPOSURE_PROFILES else "normal",
        "nombre": perfil["nombre"],
        "relacion_ac_max_orientativa": perfil["ac_max"],
        "resistencia_min_orientativa": perfil["resistencia_min"],
        "cemento_recomendado": perfil["cemento"],
        "recomendacion": perfil["recomendacion"],
        "alertas": alertas,
        "apto_orientativo": not alertas,
        "nota": "Criterio orientativo; verificar la clase de exposición, el estudio de suelo y la NTE E.060 con el responsable del diseño.",
    }

def evaluar_trabajabilidad(asentamiento_mm: float, superplasticizer: float) -> dict:
    alertas = []
    if asentamiento_mm >= 120 and superplasticizer == 0:
        alertas.append("Para un asentamiento alto sin superplastificante, validar el diseño porque aumentar agua puede reducir la resistencia.")
    return {
        "asentamiento_objetivo_mm": asentamiento_mm,
        "superplastificante_kg_m3": superplasticizer,
        "alertas": alertas,
        "control_requerido": "Verificar con ensayo de asentamiento y ficha técnica del aditivo seleccionado.",
    }

# ──────────────────────────────────────────────────────────────────
# Persistencia SQLite
# ──────────────────────────────────────────────────────────────────
DB_PATH = os.path.join(BASE_DIR, "concretoia.db")


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS material_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                unit TEXT NOT NULL,
                density REAL,
                cost_per_unit REAL,
                co2_factor REAL,
                active INTEGER NOT NULL DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS scenario_catalog (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL UNIQUE,
                name TEXT NOT NULL,
                exposure_profile TEXT NOT NULL,
                notes TEXT,
                cost_index REAL,
                co2_index REAL,
                metadata TEXT DEFAULT '{}',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS saved_compositions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                composition_json TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluations (
                id TEXT PRIMARY KEY,
                composition_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                model_name TEXT NOT NULL,
                exposure_profile TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS comparisons (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                comparison_json TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS search_history (
                id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                filters_json TEXT,
                result_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS model_versions (
                id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                version_label TEXT NOT NULL,
                filename TEXT,
                metrics_json TEXT,
                selected INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS data_versions (
                id TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                source_file TEXT,
                row_count INTEGER,
                checksum TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def seed_default_catalogs() -> None:
    conn = get_db_connection()
    try:
        material_defaults = [
            ("cement", "YURA_IP", "Cemento Yura IP", "kg/m3", 3.15, 0.42, 0.82, 1),
            ("cementitious", "SLAG_BFS", "Escoria BFS", "kg/m3", 2.85, 0.18, 0.05, 1),
            ("cementitious", "FLYASH", "Ceniza volante", "kg/m3", 2.30, 0.10, 0.004, 1),
            ("water", "WATER", "Agua", "L/m3", 1.0, 0.001, 0.001, 1),
            ("admixture", "SUPERPLASTICIZER", "Superplastificante", "kg/m3", 1.20, 2.50, 0.58, 1),
            ("aggregate", "COARSE_AGG", "Agregado grueso", "kg/m3", 1.60, 0.08, 0.048, 1),
            ("aggregate", "FINE_AGG", "Agregado fino", "kg/m3", 1.60, 0.07, 0.045, 1),
            ("admixture", "SIKA_PLAST_326", "Sika Plast-326", "kg/m3", 1.15, 3.10, 0.62, 1),
            ("admixture", "SIKA_PLASTIMENT_TM_12", "Sika Plastiment TM-12", "kg/m3", 1.15, 3.20, 0.62, 1),
        ]
        for row in material_defaults:
            category, code, name, unit, density, cost_per_unit, co2_factor, active = row
            conn.execute(
                """
                INSERT OR IGNORE INTO material_catalog (category, code, name, unit, density, cost_per_unit, co2_factor, active)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (category, code, name, unit, density, cost_per_unit, co2_factor, active),
            )

        scenario_defaults = [
            ("normal", "Sin exposición especial", "normal", "Perfil general sin estrés especial.", 1.0, 1.0),
            ("sulfatos", "Sulfatos altos en suelo o agua", "sulfatos", "Revisar estudios de suelo y resistencia a sulfatos.", 1.25, 1.15),
            ("cloruros", "Cloruros o ambiente marino", "cloruros", "Controlar permeabilidad y recubrimiento.", 1.30, 1.20),
            ("acidos", "Medio ácido", "acidos", "Solicitar evaluación especializada.", 1.45, 1.30),
        ]
        for code, name, exposure_profile, notes, cost_index, co2_index in scenario_defaults:
            conn.execute(
                """
                INSERT OR IGNORE INTO scenario_catalog (code, name, exposure_profile, notes, cost_index, co2_index)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (code, name, exposure_profile, notes, cost_index, co2_index),
            )

        dataset_path = os.path.join(BASE_DIR, "modelo", "concrete_data.csv")
        checksum = ""
        row_count = 0
        if os.path.exists(dataset_path):
            with open(dataset_path, "rb") as handle:
                checksum = hashlib.sha256(handle.read()).hexdigest()
            with open(dataset_path, "r", encoding="utf-8") as handle:
                row_count = sum(1 for _ in handle) - 1
        version_label = f"dataset-{checksum[:8]}" if checksum else "default"

        # Registro idempotente: antes cada arranque insertaba otra fila (UUID aleatorio).
        model_snapshot = {
            "modelo": selected_model_name,
            "metricas": META.get("metricas", {}),
            "benchmark": META.get("benchmark", {}),
            "features": FEATURES,
            "guardado_en": "concretoia.db",
        }
        conn.execute(
            """
            INSERT INTO model_versions (id, model_name, version_label, filename, metrics_json, selected, notes)
            SELECT ?, ?, ?, ?, ?, 1, ?
            WHERE NOT EXISTS (SELECT 1 FROM model_versions WHERE model_name = ? AND version_label = ?)
            """,
            (
                str(uuid.uuid4()), selected_model_name, version_label,
                f"{selected_model_name.lower()}.pkl",
                json.dumps(model_snapshot, ensure_ascii=False),
                f"Version registrada para {selected_model_name} entrenada con {version_label}",
                selected_model_name, version_label,
            ),
        )
        conn.execute(
            """
            INSERT INTO data_versions (id, dataset_name, source_file, row_count, checksum, notes)
            SELECT ?, ?, ?, ?, ?, ?
            WHERE NOT EXISTS (SELECT 1 FROM data_versions WHERE dataset_name = ? AND checksum = ?)
            """,
            (
                str(uuid.uuid4()), "concrete_data", dataset_path, row_count, checksum,
                "Dataset de entrenamiento (UCI Concrete Compressive Strength, Yeh 1998).",
                "concrete_data", checksum,
            ),
        )
        conn.commit()
    finally:
        conn.close()


init_db()
seed_default_catalogs()


def store_search(query: str, filters: dict, result_count: int) -> str:
    search_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO search_history (id, query, filters_json, result_count)
            VALUES (?, ?, ?, ?)
            """,
            (search_id, query, json.dumps(filters or {}, ensure_ascii=False), int(result_count)),
        )
        conn.commit()
    finally:
        conn.close()
    return search_id


def store_saved_composition(name: str, composition: dict, notes: str = "") -> str:
    comp_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO saved_compositions (id, name, composition_json, notes)
            VALUES (?, ?, ?, ?)
            """,
            (comp_id, name, json.dumps(composition, ensure_ascii=False), notes),
        )
        conn.commit()
    finally:
        conn.close()
    return comp_id


def store_evaluation(composition: dict, result: dict, model_name: str, exposure_profile: str, notes: str = "") -> str:
    eval_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO evaluations (id, composition_json, result_json, model_name, exposure_profile, notes)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                eval_id,
                json.dumps(composition, ensure_ascii=False),
                json.dumps(result, ensure_ascii=False),
                model_name,
                exposure_profile,
                notes,
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return eval_id


def store_comparison(name: str, comparison_data: list, notes: str = "") -> str:
    comp_id = str(uuid.uuid4())
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO comparisons (id, name, comparison_json, notes)
            VALUES (?, ?, ?, ?)
            """,
            (comp_id, name, json.dumps(comparison_data, ensure_ascii=False), notes),
        )
        conn.commit()
    finally:
        conn.close()
    return comp_id


class SaveCompositionIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=150)
    composition: dict
    notes: Optional[str] = ""


class SearchHistoryIn(BaseModel):
    query: str = Field(..., min_length=1, max_length=250)
    filters: dict = {}
    result_count: int = 0


class EvaluationSaveIn(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    composition: dict
    result: dict
    model_name: str = Field(default="CatBoost")
    exposure_profile: str = Field(default="normal")
    notes: Optional[str] = ""


class AssistantIn(BaseModel):
    pregunta: str = Field(..., min_length=3, max_length=1000)
    limite: int = Field(10, ge=1, le=50)

    @field_validator("pregunta")
    @classmethod
    def validar_pregunta(cls, value: str) -> str:
        cleaned = " ".join(str(value).strip().split())
        if not cleaned:
            raise ValueError("La pregunta no puede estar vacía.")
        return cleaned


# ──────────────────────────────────────────────────────────────────
# Schemas
# ──────────────────────────────────────────────────────────────────
class MezclaIn(BaseModel):
    cement:          float = Field(..., ge=100, le=600,  description="Cemento Yura IP (kg/m³)")
    slag:            float = Field(0,   ge=0,   le=400,  description="Escoria BFS (kg/m³)")
    flyash:          float = Field(0,   ge=0,   le=250,  description="Ceniza volante (kg/m³)")
    water:           float = Field(..., ge=100, le=260,  description="Agua (L/m³)")
    superplasticizer:float = Field(0,   ge=0,   le=35,   description="Superplastificante (kg/m³)")
    coarseaggregate: float = Field(..., ge=700, le=1200, description="Agregado grueso (kg/m³)")
    fineaggregate:   float = Field(..., ge=500, le=1050, description="Agregado fino (kg/m³)")
    age:             int   = Field(..., ge=1,   le=28,   description="Edad de ensayo (días)")
    exposicion_ambiental: str = Field("normal", description="normal, sulfatos, cloruros o acidos")
    asentamiento_objetivo_mm: float = Field(100, ge=0, le=250, description="Asentamiento objetivo (mm)")
    tipo_cemento: str = Field("Yura IP", max_length=80)
    aditivo_sika: str = Field("Ninguno", max_length=120)
    guardar_evaluacion: bool = False

    @field_validator("cement", "slag", "flyash", "water", "superplasticizer", "coarseaggregate", "fineaggregate", "asentamiento_objetivo_mm")
    @classmethod
    def validar_finite_numeric(cls, value: float) -> float:
        if value is None:
            return value
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Debe ser un número finito.")
        return numeric

    @field_validator("exposicion_ambiental")
    @classmethod
    def validar_exposicion(cls, value: str) -> str:
        if value is None:
            return "normal"
        normalized = str(value).strip().lower()
        if normalized not in EXPOSURE_PROFILES:
            allowed = ", ".join(sorted(EXPOSURE_PROFILES.keys()))
            raise ValueError(f"exposicion_ambiental inválida. Valores permitidos: {allowed}")
        return normalized

    @field_validator("tipo_cemento", "aditivo_sika")
    @classmethod
    def validar_texto(cls, value: str, info) -> str:
        if value is None:
            return "Ninguno" if info.field_name == "aditivo_sika" else "Yura IP"
        cleaned = str(value).strip()
        if not cleaned:
            return "Ninguno" if info.field_name == "aditivo_sika" else "Yura IP"
        return cleaned


class EvaluationReportIn(BaseModel):
    composition: MezclaIn
    result: dict = Field(default_factory=dict)
    proyecto: str = Field("Sin proyecto", max_length=150)
    responsable: str = Field("Sin responsable", max_length=150)
    fecha_generacion: Optional[str] = None

class OptimizarIn(BaseModel):
    resistencia_objetivo_kgcm2: float = Field(..., ge=100, le=700)
    agua:       float = Field(185, ge=120, le=250)
    edad:       int   = Field(28,  ge=1,   le=28)
    max_cement: float = Field(500, ge=200, le=600)
    usar_escoria: bool = False
    usar_ceniza:  bool = False
    exposicion_ambiental: str = Field("normal", description="normal, sulfatos, cloruros o acidos")
    material_ranges: dict = Field(default_factory=dict)
    objetivo_asentamiento_mm: Optional[float] = None
    objetivo_costo_sol_m3: Optional[float] = None
    objetivo_co2_kg_m3: Optional[float] = None
    prioridad: str = Field("equilibrio", pattern="^(equilibrio|costo|carbono|resistencia)$")
    n_muestras: int = Field(30000, ge=1000, le=100000, description="Mezclas candidatas evaluadas por el modelo")
    semilla: int = Field(42, description="Semilla para resultados reproducibles")
    margen_seguridad_kgcm2: Optional[float] = Field(
        None, ge=0, le=200,
        description="Margen sobre el objetivo para cubrir el error del modelo. Por defecto, el MAE del modelo activo.",
    )

    @field_validator("resistencia_objetivo_kgcm2", "agua", "max_cement")
    @classmethod
    def validar_finite_numeric(cls, value: float) -> float:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("Debe ser un número finito.")
        return numeric

    @field_validator("exposicion_ambiental")
    @classmethod
    def validar_exposicion(cls, value: str) -> str:
        if value is None:
            return "normal"
        normalized = str(value).strip().lower()
        if normalized not in EXPOSURE_PROFILES:
            allowed = ", ".join(sorted(EXPOSURE_PROFILES.keys()))
            raise ValueError(f"exposicion_ambiental inválida. Valores permitidos: {allowed}")
        return normalized

class CompararIn(BaseModel):
    mezclas: List[MezclaIn] = Field(default_factory=list)
    composicion_ids: List[str] = Field(default_factory=list, max_length=20)
    alternativas: List[dict] = Field(default_factory=list, max_length=20)
    referencia: Optional[int] = Field(default=None, ge=0, le=19)
    contexto: dict = Field(default_factory=dict)
    guardar_comparacion: bool = False
    nombre: Optional[str] = "Comparación"

    @field_validator("composicion_ids", "alternativas")
    @classmethod
    def validar_seleccion(cls, value):
        if len(value) > 20:
            raise ValueError("La comparación admite como máximo 20 composiciones.")
        return value


def _default_material_ranges() -> dict:
    return {
        "cement": {"min": 200, "max": 500, "step": 25},
        "slag": {"min": 0, "max": 250, "step": 25},
        "flyash": {"min": 0, "max": 150, "step": 25},
        "water": {"min": 120, "max": 220, "step": 10},
        "superplasticizer": {"min": 0, "max": 12, "step": 2},
        "coarseaggregate": {"min": 700, "max": 1100, "step": 50},
        "fineaggregate": {"min": 550, "max": 900, "step": 50},
        "age": {"min": 1, "max": 28, "step": 1},
    }


def _resolve_material_ranges(req_ranges: dict) -> dict:
    defaults = _default_material_ranges()
    if not isinstance(req_ranges, dict):
        return defaults
    merged = defaults.copy()
    for key, value in req_ranges.items():
        if key not in defaults:
            raise HTTPException(status_code=422, detail=f"Material no admitido en material_ranges: {key}")
        if not isinstance(value, dict):
            raise HTTPException(status_code=422, detail=f"El rango de {key} debe ser un objeto.")
        try:
            minimum = float(value.get("min", defaults[key]["min"]))
            maximum = float(value.get("max", defaults[key]["max"]))
            step = float(value.get("step", defaults[key]["step"]))
        except (TypeError, ValueError):
            raise HTTPException(status_code=422, detail=f"Los límites de {key} deben ser numéricos.")
        if not all(math.isfinite(item) for item in (minimum, maximum, step)):
            raise HTTPException(status_code=422, detail=f"Los límites de {key} deben ser finitos.")
        if minimum > maximum:
            raise HTTPException(status_code=422, detail=f"El mínimo de {key} no puede superar al máximo.")
        if step <= 0:
            raise HTTPException(status_code=422, detail=f"El incremento de {key} debe ser mayor que cero.")
        if minimum < 0 or maximum < 0:
            raise HTTPException(status_code=422, detail=f"Los límites de {key} no pueden ser negativos.")
        if maximum - minimum > 1000:
            raise HTTPException(status_code=422, detail=f"El rango de {key} es demasiado amplio.")
        if isinstance(value, dict):
            merged[key] = {
                "min": minimum,
                "max": maximum,
                "step": step,
            }
    return merged


def _reglas_perfil_tecnico(mezcla: dict, exposicion: str, asentamiento_objetivo_mm: float) -> list:
    perfil = EXPOSURE_PROFILES.get(exposicion, EXPOSURE_PROFILES["normal"])
    reglas = [
        {
            "id": 1,
            "descripcion": "Cemento principal dentro de 250 a 500 kg/m³.",
            "cumple": 250 <= float(mezcla.get("cement", 0)) <= 500,
        },
        {
            "id": 2,
            "descripcion": "Cemento + escoria + ceniza volante total entre 280 y 600 kg/m³.",
            "cumple": 280 <= float(mezcla.get("cement", 0) + mezcla.get("slag", 0) + mezcla.get("flyash", 0)) <= 600,
        },
        {
            "id": 3,
            "descripcion": f"Relación agua/cementante <= {perfil['ac_max']:.2f} para el perfil {exposicion}.",
            "cumple": relacion_agua_cementante(mezcla.get("cement", 0), mezcla.get("slag", 0), mezcla.get("flyash", 0), mezcla.get("water", 0)) <= perfil["ac_max"],
        },
        {
            "id": 4,
            "descripcion": "Agua en rango de 120 a 220 L/m³.",
            "cumple": 120 <= float(mezcla.get("water", 0)) <= 220,
        },
        {
            "id": 5,
            "descripcion": "Superplastificante no supera 12 kg/m³ en mezclas normales y 20 kg/m³ en altas trabajabilidades.",
            "cumple": float(mezcla.get("superplasticizer", 0)) <= (20 if float(asentamiento_objetivo_mm) >= 120 else 12),
        },
        {
            "id": 6,
            "descripcion": "Agregado grueso en 700 a 1100 kg/m³.",
            "cumple": 700 <= float(mezcla.get("coarseaggregate", 0)) <= 1100,
        },
        {
            "id": 7,
            "descripcion": "Agregado fino en 550 a 900 kg/m³.",
            "cumple": 550 <= float(mezcla.get("fineaggregate", 0)) <= 900,
        },
        {
            "id": 8,
            "descripcion": "Si el asentamiento objetivo es alto, se debe usar superplastificante activo.",
            "cumple": not (float(asentamiento_objetivo_mm) >= 120 and float(mezcla.get("superplasticizer", 0)) <= 0),
        },
        {
            "id": 9,
            "descripcion": "Edad de ensayo entre 1 y 28 días (rango admitido por la API).",
            "cumple": 1 <= int(mezcla.get("age", 28)) <= 28,
        },
    ]
    return reglas


def _evaluar_restricciones(req: OptimizarIn, mezcla: dict, pred_kgcm2: float, asentamiento_objetivo_mm: float) -> dict:
    reglas = _reglas_perfil_tecnico(mezcla, req.exposicion_ambiental, asentamiento_objetivo_mm)
    incumplidas = [r for r in reglas if not r["cumple"]]
    return {
        "cumple_todas": not incumplidas,
        "reglas": reglas,
        "restricciones_incumplidas": [r["descripcion"] for r in incumplidas],
    }


def _pareto_frontier(candidatos: list) -> list:
    """Frontera de Pareto costo vs CO2 (ambos a minimizar) entre candidatos ya factibles.

    La resistencia es una restricción (debe superar el objetivo), no un objetivo a
    maximizar: pasarse del objetivo solo encarece la mezcla. Barrido O(n log n).
    """
    ordenados = sorted(candidatos, key=lambda x: (x["costo_sol_m3"], x["co2_kg_m3"]))
    frontera, mejor_co2 = [], float("inf")
    for item in ordenados:
        if item["co2_kg_m3"] < mejor_co2 - 1e-9:
            frontera.append(item)
            mejor_co2 = item["co2_kg_m3"]
    return frontera


ESTRATEGIAS = {
    "equilibrio": "Mejor equilibrio costo-CO2",
    "costo": "Más económica",
    "carbono": "Menor huella de carbono",
    "resistencia": "Más cercana al objetivo",
}


def _sample_grid(rng: np.random.Generator, spec: dict, n: int) -> np.ndarray:
    values = np.arange(spec["min"], spec["max"] + spec["step"] * 0.5, spec["step"])
    values = values[values <= spec["max"] + 1e-9]
    return values[rng.integers(0, len(values), size=n)].astype(float)


def _run_optimization(req: OptimizarIn, advanced: bool = False):
    """Búsqueda de mezclas que cumplen el objetivo de resistencia con menor costo/CO2.

    1. Muestrea ``n_muestras`` dosificaciones sobre una grilla (respetando agua, edad,
       cemento máximo y uso de escoria/ceniza solicitados).
    2. Predice resistencia de todas en un solo lote con el modelo activo.
    3. Filtra las que cumplen las 9 reglas técnicas, la durabilidad del perfil y
       el dominio de datos con que se entrenó el modelo.
    4. Ordena según la prioridad pedida y devuelve alternativas distintas entre sí,
       la frontera de Pareto y el ahorro frente a la mejor mezcla sin adiciones.
    """
    perfil = EXPOSURE_PROFILES[req.exposicion_ambiental]
    objetivo_usuario = float(req.resistencia_objetivo_kgcm2)
    objetivo = max(objetivo_usuario, float(perfil["resistencia_min"]))
    asentamiento = req.objetivo_asentamiento_mm if req.objetivo_asentamiento_mm is not None else 100.0
    objetivo_costo = req.objetivo_costo_sol_m3
    objetivo_co2 = req.objetivo_co2_kg_m3
    margen = (
        float(req.margen_seguridad_kgcm2)
        if req.margen_seguridad_kgcm2 is not None
        else float(META.get("metricas", {}).get("mae_mpa", 0.0)) * 10.197
    )
    meta_resistencia = objetivo + margen

    rangos = _resolve_material_ranges(req.material_ranges)
    pedidos = req.material_ranges if isinstance(req.material_ranges, dict) else {}
    # El usuario fija agua y edad salvo que pida explícitamente explorarlas.
    if "water" not in pedidos:
        rangos["water"] = {"min": float(req.agua), "max": float(req.agua), "step": 1.0}
    if "age" not in pedidos:
        rangos["age"] = {"min": float(req.edad), "max": float(req.edad), "step": 1.0}
    if "superplasticizer" not in pedidos and asentamiento >= 120:
        rangos["superplasticizer"] = {"min": 0.0, "max": 20.0, "step": 2.0}
    if not req.usar_escoria and "slag" not in pedidos:
        rangos["slag"] = {"min": 0.0, "max": 0.0, "step": 1.0}
    if not req.usar_ceniza and "flyash" not in pedidos:
        rangos["flyash"] = {"min": 0.0, "max": 0.0, "step": 1.0}
    rangos["cement"]["max"] = min(rangos["cement"]["max"], float(req.max_cement))
    if rangos["cement"]["min"] > rangos["cement"]["max"]:
        raise HTTPException(status_code=422, detail="El cemento mínimo supera el cemento máximo permitido (max_cement).")

    n = int(req.n_muestras)
    rng = np.random.default_rng(req.semilla)
    cols = {f: _sample_grid(rng, rangos[f], n) for f in FEATURES}
    X = np.column_stack([cols[f] for f in FEATURES])

    pred_mpa = np.asarray(model.predict(scaler.transform(X)), dtype=float)
    pred_kgcm2 = pred_mpa * 10.197

    cement, slag, flyash, water = cols["cement"], cols["slag"], cols["flyash"], cols["water"]
    superp, coarse, fine, age = cols["superplasticizer"], cols["coarseaggregate"], cols["fineaggregate"], cols["age"]
    cementante = cement + slag + flyash
    ac = water / (cementante + 1e-9)
    co2 = sum(cols[k] * CO2_FACTOR[k] for k in CO2_FACTOR)
    costo = sum(cols[k] * COST_FACTOR[k] for k in COST_FACTOR)
    adicion = np.where(cementante > 0, (slag + flyash) / np.maximum(cementante, 1e-9) * 100.0, 0.0)

    # Mismas 9 reglas que _reglas_perfil_tecnico, en forma vectorizada.
    reglas_ok = (
        (cement >= 250) & (cement <= 500)
        & (cementante >= 280) & (cementante <= 600)
        & (ac <= perfil["ac_max"])
        & (water >= 120) & (water <= 220)
        & (superp <= (20 if asentamiento >= 120 else 12))
        & (coarse >= 700) & (coarse <= 1100)
        & (fine >= 550) & (fine <= 900)
        & ~((asentamiento >= 120) & (superp <= 0))
        & (age >= 1) & (age <= 28)
    )
    en_dominio = np.ones(n, dtype=bool)
    for idx, f in enumerate(FEATURES):
        bounds = TRAINING_DOMAIN.get(f)
        if bounds:
            en_dominio &= (X[:, idx] >= bounds[0] - 1e-9) & (X[:, idx] <= bounds[1] + 1e-9)

    cumple_objetivo = pred_kgcm2 >= meta_resistencia
    factibles = reglas_ok & en_dominio & cumple_objetivo
    sobrepaso = np.maximum(pred_kgcm2 - meta_resistencia, 0.0)

    def _norm(v: np.ndarray, mask: np.ndarray) -> np.ndarray:
        sel = v[mask]
        span = float(sel.max() - sel.min()) if sel.size else 0.0
        return (v - (sel.min() if sel.size else 0.0)) / span if span > 0 else np.zeros_like(v)

    def _clave(prioridad: str, mask: np.ndarray) -> np.ndarray:
        if prioridad == "costo":
            base = costo + 1e-3 * co2
        elif prioridad == "carbono":
            base = co2 + 1e-3 * costo
        elif prioridad == "resistencia":
            base = sobrepaso + 1e-3 * costo
        else:
            base = 0.5 * _norm(costo, mask) + 0.5 * _norm(co2, mask) + 0.05 * _norm(sobrepaso, mask)
        extra = np.zeros(n)
        if objetivo_costo is not None:
            extra += np.abs(costo - objetivo_costo) / max(objetivo_costo, 1.0) * 0.25
        if objetivo_co2 is not None:
            extra += np.abs(co2 - objetivo_co2) / max(objetivo_co2, 1.0) * 0.25
        if prioridad == "equilibrio":
            return base + extra
        return base * (1 + extra)

    def _top(prioridad: str, mask: np.ndarray, k: int) -> list:
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        orden = idx[np.argsort(_clave(prioridad, mask)[idx], kind="stable")]
        return list(orden[:k])

    def _candidato(i: int, estrategia: str, puntaje: float) -> dict:
        mezcla = {f: float(cols[f][i]) for f in FEATURES}
        mezcla["age"] = int(mezcla["age"])
        kg = float(pred_kgcm2[i])
        restricciones = _evaluar_restricciones(req, mezcla, kg, asentamiento)
        diff = abs(kg - objetivo_usuario) / objetivo_usuario * 100 if objetivo_usuario else 0.0
        return {
            **mezcla,
            "resistencia_mpa": round(float(pred_mpa[i]), 3),
            "resistencia_kgcm2": round(kg, 2),
            "clasificacion": clasificar(kg),
            "co2_kg_m3": round(float(co2[i]), 2),
            "costo_sol_m3": round(float(costo[i]), 2),
            "relacion_ac": round(float(ac[i]), 3),
            "adicion_pct": round(float(adicion[i]), 1),
            "diferencia_pct": round(diff, 2),
            "cumple_objetivo": bool(cumple_objetivo[i]),
            "fuera_de_dominio": bool(not en_dominio[i]),
            "durabilidad": evaluar_exposicion(req.exposicion_ambiental, kg, float(ac[i])),
            "cumple_reglas": restricciones["cumple_todas"],
            "restricciones_incumplidas": restricciones["restricciones_incumplidas"],
            "reglas": restricciones["reglas"],
            "estrategia": estrategia,
            "score": round(float(puntaje), 4),
        }

    advertencias = []
    if objetivo > objetivo_usuario:
        advertencias.append(
            f"El perfil '{req.exposicion_ambiental}' exige al menos {objetivo:.0f} kg/cm²; se optimizó para ese valor."
        )
    factible = bool(factibles.any())
    if factible:
        mascara = factibles
        etiqueta = ESTRATEGIAS
    else:
        # Sin mezclas que cumplan todo: se muestran las más cercanas para orientar el ajuste.
        base = reglas_ok & en_dominio
        mascara = base if base.any() else en_dominio
        if not mascara.any():
            raise HTTPException(status_code=404, detail="Ninguna mezcla de los rangos indicados cae dentro del dominio del modelo.")
        advertencias.append(
            f"Ninguna mezcla alcanza {meta_resistencia:.0f} kg/cm² (objetivo + margen del modelo de {margen:.0f}) "
            "cumpliendo reglas y durabilidad. Se muestran las más resistentes; relaje cemento máximo, agua o use adiciones."
        )
        etiqueta = {k: "Más cercana al objetivo" for k in ESTRATEGIAS}

    max_res = float(TRAINING_DOMAIN.get("_resistencia_max_mpa", 0.0)) * 10.197
    if max_res and meta_resistencia > max_res:
        advertencias.append(
            f"El objetivo supera la resistencia máxima del dataset de entrenamiento ({max_res:.0f} kg/cm²): "
            "la predicción no es confiable en ese rango."
        )

    claves = {p: _clave(p, mascara) for p in ESTRATEGIAS}
    if not factible:
        orden = np.flatnonzero(mascara)
        orden = orden[np.argsort(-pred_kgcm2[orden], kind="stable")]
        elegidos = [(int(i), "Más cercana al objetivo") for i in orden[:60]]
    else:
        elegidos = []
        orden_estrategias = [req.prioridad] + [p for p in ESTRATEGIAS if p != req.prioridad]
        for p in orden_estrategias:
            top = _top(p, mascara, 1)
            if top:
                elegidos.append((int(top[0]), etiqueta[p]))
        for i in _top(req.prioridad, mascara, int(mascara.sum())):
            elegidos.append((int(i), "Alternativa"))

    # Alternativas distintas entre sí (misma terna cemento/escoria/ceniza = casi la misma mezcla).
    vistos, alternativas = set(), []
    for i, estrategia in elegidos:
        firma = (cement[i], slag[i], flyash[i])
        if firma in vistos:
            continue
        vistos.add(firma)
        alternativas.append(_candidato(i, estrategia, claves[req.prioridad][i]))
        if len(alternativas) == 10:
            break

    frontera = []
    if factible:
        idx = np.flatnonzero(mascara)
        pool = [_candidato(int(i), "Frontera de Pareto", claves["equilibrio"][i]) for i in idx[np.argsort(costo[idx])][:2000]]
        frontera = _pareto_frontier(pool)[:10]

    # Ahorro frente a la mejor mezcla convencional (sin escoria ni ceniza) del mismo muestreo.
    convencional = None
    sin_adicion = mascara & (slag == 0) & (flyash == 0)
    if factible and sin_adicion.any():
        top = _top(req.prioridad, sin_adicion, 1)
        if top:
            convencional = _candidato(int(top[0]), "Convencional (sin adiciones)", claves[req.prioridad][top[0]])
    if convencional:
        for alt in alternativas:
            alt["ahorro_co2_vs_convencional_kg_m3"] = round(convencional["co2_kg_m3"] - alt["co2_kg_m3"], 2)
            alt["ahorro_costo_vs_convencional_sol_m3"] = round(convencional["costo_sol_m3"] - alt["costo_sol_m3"], 2)
            alt["reduccion_co2_pct"] = round(
                (convencional["co2_kg_m3"] - alt["co2_kg_m3"]) / convencional["co2_kg_m3"] * 100, 1
            ) if convencional["co2_kg_m3"] else 0.0

    resumen = {
        "factible": factible,
        "objetivo_resistencia_kgcm2": objetivo_usuario,
        "objetivo_resistencia_efectivo_kgcm2": round(objetivo, 1),
        "margen_seguridad_kgcm2": round(margen, 1),
        "objetivo_asentamiento_mm": asentamiento,
        "objetivo_costo_sol_m3": objetivo_costo,
        "objetivo_co2_kg_m3": objetivo_co2,
        "prioridad": req.prioridad,
        "perfil": req.exposicion_ambiental,
        "reglas_obligatorias": 9,
        "mezclas_evaluadas": n,
        "mezclas_factibles": int(factibles.sum()),
        "cumplen_reglas": sum(1 for item in alternativas if item["cumple_reglas"]),
        "restricciones_incumplidas_totales": sorted({rule for item in alternativas for rule in item["restricciones_incumplidas"]}),
        "advertencias": advertencias,
        "referencia_convencional": convencional,
        "alternativas": alternativas,
        "frontera_pareto": frontera,
        "seleccion": alternativas[:3],
        "modelo": selected_model_name,
        "nota": "Predicción estadística del modelo; validar con ensayos de laboratorio y el responsable estructural.",
    }
    if advanced:
        resumen["material_ranges"] = rangos
    return resumen


class ModeloSeleccionarIn(BaseModel):
    modelo: str

    @field_validator("modelo")
    @classmethod
    def validar_modelo(cls, value: str) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("modelo requerido")
        return cleaned


@app.get("/catalogo/materiales", tags=["Catálogo"])
def listar_materiales():
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM material_catalog ORDER BY category, name"
        ).fetchall()
        return {"materiales": [dict(row) for row in rows]}
    finally:
        conn.close()


@app.post("/catalogo/materiales", tags=["Catálogo"])
def crear_material(material: dict):
    required = ["category", "code", "name", "unit"]
    missing = [field for field in required if field not in material or material[field] in (None, "")]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan campos obligatorios: {', '.join(missing)}")
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO material_catalog (category, code, name, unit, density, cost_per_unit, co2_factor, active, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                category=excluded.category,
                name=excluded.name,
                unit=excluded.unit,
                density=excluded.density,
                cost_per_unit=excluded.cost_per_unit,
                co2_factor=excluded.co2_factor,
                active=excluded.active,
                metadata=excluded.metadata
            """,
            (
                str(material.get("category", "material")).strip(),
                str(material["code"]).strip(),
                str(material["name"]).strip(),
                str(material["unit"]).strip(),
                float(material.get("density", 0) or 0),
                float(material.get("cost_per_unit", 0) or 0),
                float(material.get("co2_factor", 0) or 0),
                int(bool(material.get("active", True))),
                json.dumps(material.get("metadata", {}), ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"estado": "guardado"}


@app.get("/catalogo/escenarios", tags=["Catálogo"])
def listar_escenarios():
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM scenario_catalog ORDER BY id"
        ).fetchall()
        return {"escenarios": [dict(row) for row in rows]}
    finally:
        conn.close()


@app.post("/catalogo/escenarios", tags=["Catálogo"])
def crear_escenario(escenario: dict):
    required = ["code", "name", "exposure_profile"]
    missing = [field for field in required if field not in escenario or escenario[field] in (None, "")]
    if missing:
        raise HTTPException(status_code=422, detail=f"Faltan campos obligatorios: {', '.join(missing)}")
    conn = get_db_connection()
    try:
        conn.execute(
            """
            INSERT INTO scenario_catalog (code, name, exposure_profile, notes, cost_index, co2_index, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(code) DO UPDATE SET
                name=excluded.name,
                exposure_profile=excluded.exposure_profile,
                notes=excluded.notes,
                cost_index=excluded.cost_index,
                co2_index=excluded.co2_index,
                metadata=excluded.metadata
            """,
            (
                str(escenario["code"]).strip(),
                str(escenario["name"]).strip(),
                str(escenario["exposure_profile"]).strip(),
                escenario.get("notes") or "",
                float(escenario.get("cost_index", 1.0) or 1.0),
                float(escenario.get("co2_index", 1.0) or 1.0),
                json.dumps(escenario.get("metadata", {}), ensure_ascii=False),
            ),
        )
        conn.commit()
    finally:
        conn.close()
    return {"estado": "guardado"}


@app.get("/composiciones", tags=["Persistencia"])
def listar_composiciones(limit: int = Query(50, ge=1, le=200)):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM saved_compositions ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"composiciones": [dict(row) for row in rows]}
    finally:
        conn.close()


@app.post("/importar/pdf", tags=["Importación"])
async def importar_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Debes seleccionar un archivo PDF.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="El PDF supera el límite de 10 MB.")

    try:
        reader = PdfReader(BytesIO(contents))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"No se pudo leer el PDF: {exc}")

    def find_number(patterns: List[str]) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                raw = match.group(1).strip().replace(" ", "")
                if "," in raw and "." in raw:
                    raw = raw.replace(".", "") if raw.rfind(".") < raw.rfind(",") else raw.replace(",", "")
                raw = raw.replace(",", ".")
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    def find_text(pattern: str) -> Optional[str]:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        return match.group(1).strip() if match else None

    section_match = re.search(
        r"Cantidades\s+del\s+diseño\s+por\s*m\s*[³3].*?(?=Dosificación\s+por\s+bolsa|Granulometría|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    design_section = section_match.group(0) if section_match else text

    def find_in_design(patterns: List[str]) -> Optional[float]:
        for pattern in patterns:
            match = re.search(pattern, design_section, flags=re.IGNORECASE | re.MULTILINE)
            if match:
                raw = match.group(1).strip().replace(" ", "")
                if "," in raw and "." in raw:
                    raw = raw.replace(".", "") if raw.rfind(".") < raw.rfind(",") else raw.replace(",", "")
                raw = raw.replace(",", ".")
                try:
                    return float(raw)
                except ValueError:
                    continue
        return None

    datos = {
        "cement": find_in_design([
            r"Cemento(?:\s+Portland)?[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"(?<!\w)Cemento\b.{0,80}?([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "water": find_in_design([
            r"(?:^|\n)Agua(?:\s+efectiva)?[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Agua\b.{0,80}?([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "coarseaggregate": find_in_design([
            r"(?:^|\n)Agregado\s+grueso[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Agregado\s+grueso.{0,80}?([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "fineaggregate": find_in_design([
            r"(?:^|\n)Agregado\s+fino[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Agregado\s+fino.{0,80}?([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "asentamiento_objetivo_mm": find_number([
            r"Asentamiento\s+de\s+diseño[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Asentamiento\s+de\s+diseno[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Asentamiento[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "unidad_asentamiento": "in",
        "proyecto": find_text(r"(?:Nombre|Proyecto)[\s:]+([^\n]+)") or "",
        "metodo_diseno": find_text(r"Método\s+de\s+diseño[\s:]+([^\n]+)") or "",
        "tamano_maximo_nominal_mm": find_number([
            r"Tamaño\s+máximo\s+nominal[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Tamano\s+maximo\s+nominal[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "tamano_maximo_curva_mm": find_number([
            r"Tamaño\s+máximo\s+para\s+la\s+curva[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
            r"Tamano\s+maximo\s+para\s+la\s+curva[^0-9]*([0-9]+(?:[.,][0-9]+)?)",
        ]),
        "malla_referencia_mm": find_number([r"Malla\s+de\s+referencia[^0-9]*([0-9]+(?:[.,][0-9]+)?)"]),
        "k1": find_number([r"(?<!\w)K1[^0-9]*([0-9]+(?:[.,][0-9]+)?)"]),
        "coeficiente_curva": find_number([r"Coeficiente\s+de\s+la\s+curva[^0-9]*([0-9]+(?:[.,][0-9]+)?)"]),
        "resistencia_solicitada_kgcm2": find_number([r"Resistencia\s+solicitada[^0-9]*([0-9]+(?:[.,][0-9]+)?)"]),
        "responsable": find_text(r"Responsable(?:\s+técnico)?[\s:]+([^\n]+)") or "",
    }

    if datos["asentamiento_objetivo_mm"] is not None:
        asentamiento_contexto = re.search(
            r"Asentamiento.{0,100}([0-9]+(?:[.,][0-9]+)?)[^\n]*(in\b|inch|pulgadas?)|([0-9]+(?:[.,][0-9]+)?)[^\n]*(in\b|inch|pulgadas?).{0,100}Asentamiento",
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if asentamiento_contexto:
            datos["asentamiento_objetivo_mm"] = round(datos["asentamiento_objetivo_mm"] * 25.4, 1)
            datos["unidad_asentamiento"] = "mm"
        else:
            datos["unidad_asentamiento"] = "mm"

    for name in ("cement", "water", "coarseaggregate", "fineaggregate"):
        if datos[name] is not None:
            datos[name] = round(datos[name], 2)

    for name in ("tamano_maximo_nominal_mm", "tamano_maximo_curva_mm", "malla_referencia_mm", "k1", "coeficiente_curva", "resistencia_solicitada_kgcm2"):
        if datos[name] is not None:
            datos[name] = round(datos[name], 5)

    reconocidos = [name for name, value in datos.items() if value is not None and value != ""]
    if not reconocidos:
        raise HTTPException(status_code=422, detail="No encontré cantidades de mezcla compatibles en el PDF.")

    return {"datos": datos, "reconocidos": reconocidos, "texto_pagina": text[:500]}

@app.post("/composiciones", tags=["Persistencia"])
def guardar_composicion(payload: SaveCompositionIn):
    comp_id = store_saved_composition(payload.name, payload.composition, payload.notes or "")
    return {"id": comp_id, "estado": "guardado"}


@app.get("/evaluaciones", tags=["Persistencia"])
def listar_evaluaciones(limit: int = Query(50, ge=1, le=200)):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM evaluations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"evaluaciones": [dict(row) for row in rows]}
    finally:
        conn.close()


@app.get("/evaluaciones/{evaluation_id}", tags=["Persistencia"])
def obtener_evaluacion(evaluation_id: str):
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM evaluations WHERE id = ?",
            (evaluation_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Evaluación no encontrada")
        payload = dict(row)
        payload["composition"] = json.loads(payload.pop("composition_json"))
        payload["result"] = json.loads(payload.pop("result_json"))
        return payload
    finally:
        conn.close()


@app.post("/evaluaciones", tags=["Persistencia"])
def guardar_evaluacion(payload: EvaluationSaveIn):
    eval_id = store_evaluation(payload.composition, payload.result, payload.model_name, payload.exposure_profile, payload.notes or "")
    return {"id": eval_id, "estado": "guardado"}


@app.get("/busquedas", tags=["Persistencia"])
def listar_busquedas(limit: int = Query(50, ge=1, le=200)):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM search_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"busquedas": [dict(row) for row in rows]}
    finally:
        conn.close()


@app.post("/busquedas", tags=["Persistencia"])
def guardar_busqueda(payload: SearchHistoryIn):
    search_id = store_search(payload.query, payload.filters or {}, payload.result_count)
    return {"id": search_id, "estado": "guardado"}


@app.get("/comparaciones", tags=["Persistencia"])
def listar_comparaciones(limit: int = Query(50, ge=1, le=200)):
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT id, name, notes, created_at FROM comparisons ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return {"comparaciones": [dict(row) for row in rows]}
    finally:
        conn.close()


def _json_or_default(value: Optional[str], default):
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def _assistant_records(question: str, limit: int) -> list:
    terms = [term for term in question.lower().split() if len(term) >= 3]
    conn = get_db_connection()
    try:
        queries = [
            ("búsqueda", "search_history", "query, filters_json, result_count, created_at"),
            ("evaluación", "evaluations", "id, composition_json, result_json, model_name, exposure_profile, created_at"),
            ("comparación", "comparisons", "id, name, comparison_json, notes, created_at"),
        ]
        candidates = []
        for source_type, table, columns in queries:
            rows = conn.execute(f"SELECT {columns} FROM {table} ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            for row in rows:
                record = dict(row)
                haystack = json.dumps(record, ensure_ascii=False).lower()
                score = sum(haystack.count(term) for term in terms)
                candidates.append((score, record["created_at"], source_type, record))
        candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected = candidates[:limit]
        output = []
        for score, _, source_type, record in selected:
            if source_type == "evaluación":
                output.append({
                    "tipo": source_type,
                    "id": record["id"],
                    "fecha": record["created_at"],
                    "composicion": _json_or_default(record["composition_json"], {}),
                    "resultado": _json_or_default(record["result_json"], {}),
                    "modelo": record["model_name"],
                    "exposicion": record["exposure_profile"],
                    "coincidencias": score,
                })
            elif source_type == "comparación":
                comparison = _json_or_default(record["comparison_json"], {})
                if isinstance(comparison, list):
                    comparison = {"resultados": comparison}
                if not isinstance(comparison, dict):
                    comparison = {}
                output.append({
                    "tipo": source_type,
                    "id": record["id"],
                    "nombre": record["name"],
                    "fecha": record["created_at"],
                    "referencia": comparison.get("referencia_nombre"),
                    "resultados": comparison.get("resultados", []),
                    "contexto_comun": comparison.get("contexto_comun", {}),
                    "coincidencias": score,
                })
            else:
                output.append({
                    "tipo": source_type,
                    "fecha": record["created_at"],
                    "consulta": record["query"],
                    "filtros": _json_or_default(record["filters_json"], {}),
                    "resultados": record["result_count"],
                    "coincidencias": score,
                })
        return output
    finally:
        conn.close()


@app.post("/asistente/consultar", tags=["Asistente IA"])
def consultar_asistente(payload: AssistantIn):
    records = _assistant_records(payload.pregunta, payload.limite)
    if not records:
        answer = (
            "No encontré búsquedas, evaluaciones ni comparaciones guardadas. "
            "La respuesta no se puede fundamentar con datos del sistema."
        )
    else:
        evaluations = [item for item in records if item["tipo"] == "evaluación"]
        comparisons = [item for item in records if item["tipo"] == "comparación"]
        searches = [item for item in records if item["tipo"] == "búsqueda"]
        answer_parts = [f"Encontré {len(records)} registros guardados relacionados con la consulta."]
        if evaluations:
            valid = [item for item in evaluations if isinstance(item.get("resultado"), dict)]
            resistances = [
                (item["resultado"].get("resistencia_kgcm2"), item["id"])
                for item in valid
                if isinstance(item["resultado"].get("resistencia_kgcm2"), (int, float))
            ]
            if resistances:
                best = max(resistances, key=lambda item: item[0])
                answer_parts.append(f"La mayor resistencia registrada es {best[0]:.2f} kg/cm² (evaluación {best[1]}).")
            else:
                answer_parts.append(f"Hay {len(evaluations)} evaluaciones, sin una resistencia numérica disponible.")
        if comparisons:
            answer_parts.append(f"Hay {len(comparisons)} comparaciones con contexto y referencia persistidos.")
        if searches:
            answer_parts.append(f"Hay {len(searches)} búsquedas registradas, incluyendo sus filtros y cantidad de resultados.")
        answer_parts.append("Las afirmaciones anteriores se basan únicamente en los registros incluidos en 'fuentes'.")
        answer = " ".join(answer_parts)
    return {
        "respuesta": answer,
        "pregunta": payload.pregunta,
        "fuentes": records,
        "fuentes_consultadas": ["busquedas", "evaluaciones", "comparaciones"],
        "limitado": len(records) >= payload.limite,
    }


@app.get("/modelos/versiones", tags=["Persistencia"])
def listar_modelos_versiones():
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM model_versions ORDER BY created_at DESC"
        ).fetchall()
        return {"versiones": [dict(row) for row in rows]}
    finally:
        conn.close()


@app.get("/datos/versiones", tags=["Persistencia"])
def listar_versiones_datos():
    conn = get_db_connection()
    try:
        rows = conn.execute(
            "SELECT * FROM data_versions ORDER BY created_at DESC"
        ).fetchall()
        return {"versiones": [dict(row) for row in rows]}
    finally:
        conn.close()


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def raiz():
    return {
        "app": "ConcretoIA",
        "version": "1.0.0",
        "empresa": "Cemento Yura S.A.",
        "hackathon": "Innova Sur 2026 — Reto 2",
        "docs": "/docs",
    }

@app.get("/salud", tags=["Info"])
def salud():
    return {
        "estado": "ok",
        "modelo": "cargado" if MODEL_OK else "no disponible",
    }

@app.get("/modelo/info", tags=["Modelo"])
def modelo_info():
    available = list(AVAILABLE_MODELS)
    benchmark = META.get("benchmark", {})
    active_metrics = benchmark.get(selected_model_name)
    return {
        **META,
        "modelo_activo": selected_model_name,
        "modelos_disponibles": available,
        "modelo_cargado": MODEL_OK,
        "metricas_activo": active_metrics,
        "metricas": active_metrics or META.get("metricas", {}),
    }

@app.post("/modelo/seleccionar", tags=["Modelo"])
def seleccionar_modelo(req: ModeloSeleccionarIn):
    global model, selected_model_name, MODEL_OK
    if req.modelo not in AVAILABLE_MODELS:
        raise HTTPException(status_code=404, detail="Modelo no disponible. Ejecuta nuevamente train_model.py.")
    model = AVAILABLE_MODELS[req.modelo]
    selected_model_name = req.modelo
    MODEL_OK = model is not None and scaler is not None and _is_scaler_compatible(scaler) and _is_model_compatible(model)
    if not MODEL_OK:
        raise HTTPException(status_code=503, detail="El modelo seleccionado no es compatible con el scaler cargado.")
    return {"modelo_activo": selected_model_name, "estado": "seleccionado"}

@app.post("/predecir", tags=["Predicción"])
def predecir(m: MezclaIn):
    require_model_ready()
    data = m.model_dump()
    missing_fields = _missing_input_fields(data)
    inp = build_feature_vector(data)
    inp_scaled = scaler.transform(inp)
    mpa = float(model.predict(inp_scaled)[0])
    kgcm2 = mpa * 10.197

    co2, cost = calc_extras(data)
    ac = relacion_agua_cementante(m.cement, m.slag, m.flyash, m.water)
    durabilidad = evaluar_exposicion(m.exposicion_ambiental, kgcm2, ac)
    trabajabilidad = evaluar_trabajabilidad(m.asentamiento_objetivo_mm, m.superplasticizer)
    desglose = _breakdown_cost_and_carbon(data)
    result = {
        "resistencia_mpa":   round(mpa, 3),
        "resistencia_kgcm2": round(kgcm2, 2),
        "clasificacion":     clasificar(kgcm2),
        "co2_kg_m3":         co2,
        "costo_sol_m3":      cost,
        "relacion_ac":       round(ac, 3),
        "cumple_nte_e060":   ac <= 0.65 and kgcm2 >= 210,
        "durabilidad":       durabilidad,
        "trabajabilidad":    trabajabilidad,
        "materiales": {
            "tipo_cemento": m.tipo_cemento,
            "aditivo_sika": m.aditivo_sika,
        },
        "datos_faltantes": missing_fields,
        "desglose_costos": desglose["costos_por_material"],
        "desglose_co2": desglose["co2_por_material"],
        "verificaciones_tecnicas": {
            "ac_ok": ac <= 0.65,
            "resistencia_minima_ok": kgcm2 >= 210,
            "cumple_nte_e060": ac <= 0.65 and kgcm2 >= 210,
            "perfil_ambiental": m.exposicion_ambiental,
            "alertas_durabilidad": durabilidad.get("alertas", []),
            "alertas_trabajabilidad": trabajabilidad.get("alertas", []),
        },
        "procedencia": {
            "fuente": "manual",
            "modelo": selected_model_name,
            "proyecto": getattr(m, "proyecto", "") or "Sin proyecto",
            "responsable": getattr(m, "responsable", "") or "Sin responsable",
            "exposicion_ambiental": m.exposicion_ambiental,
            "edad_dias": m.age,
        },
    }
    if m.guardar_evaluacion:
        eval_id = store_evaluation(data, result, selected_model_name, m.exposicion_ambiental, "Predicción individual")
        result["persistencia"] = {"evaluacion_id": eval_id, "guardada": True}
    return result

@app.post("/optimizar", tags=["Optimización"])
def optimizar(req: OptimizarIn):
    require_model_ready()
    return _run_optimization(req, advanced=True)

@app.post("/comparar", tags=["Comparación"])
def comparar(req: CompararIn):
    require_model_ready()

    selections = []
    for mixture in req.mezclas:
        selections.append({"name": f"Mezcla {len(selections) + 1}", "source": "inline", "data": mixture.model_dump()})

    conn = get_db_connection()
    try:
        for composition_id in req.composicion_ids:
            row = conn.execute(
                "SELECT id, name, composition_json FROM saved_compositions WHERE id = ?",
                (composition_id,),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Composición guardada no encontrada: {composition_id}")
            try:
                composition = json.loads(row["composition_json"])
            except json.JSONDecodeError as exc:
                raise HTTPException(status_code=422, detail=f"La composición guardada {composition_id} no contiene JSON válido.") from exc
            selections.append({"name": row["name"], "source": "guardada", "id": row["id"], "data": composition})
    finally:
        conn.close()

    for index, alternative in enumerate(req.alternativas):
        if not isinstance(alternative, dict):
            raise HTTPException(status_code=422, detail="Cada alternativa debe ser un objeto JSON.")
        data = alternative.get("composicion", alternative.get("composition", alternative))
        if not isinstance(data, dict):
            raise HTTPException(status_code=422, detail=f"La alternativa {index + 1} no contiene una composición.")
        selections.append({
            "name": str(alternative.get("nombre", alternative.get("name", f"Alternativa {index + 1}"))),
            "source": "alternativa",
            "data": data,
        })

    if not 2 <= len(selections) <= 20:
        raise HTTPException(status_code=422, detail="La comparación debe incluir entre 2 y 20 composiciones.")
    if req.referencia is not None and req.referencia >= len(selections):
        raise HTTPException(status_code=422, detail="La referencia indicada no existe en la selección.")

    context = dict(req.contexto or {})
    allowed_context = {"exposicion_ambiental", "asentamiento_objetivo_mm", "age", "tipo_cemento", "aditivo_sika"}
    unknown_context = sorted(set(context) - allowed_context)
    if unknown_context:
        raise HTTPException(status_code=422, detail=f"Contexto no admitido: {', '.join(unknown_context)}")
    reference_source = selections[req.referencia or 0]["data"]
    for key in allowed_context:
        if key not in context and reference_source.get(key) is not None:
            context[key] = reference_source[key]
    out = []
    for i, selection in enumerate(selections):
        data = dict(selection["data"])
        data.update(context)
        try:
            mixture = MezclaIn.model_validate(data)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"Composición '{selection['name']}' inválida: {exc}") from exc
        data = mixture.model_dump()
        inp = build_feature_vector(data)
        inp_scaled = scaler.transform(inp)
        mpa = float(model.predict(inp_scaled)[0])
        kgcm2 = mpa * 10.197
        co2, cost = calc_extras(data)
        ac = relacion_agua_cementante(mixture.cement, mixture.slag, mixture.flyash, mixture.water)
        out.append({
            "id": i + 1,
            "nombre": selection["name"],
            "fuente": selection["source"],
            "composicion_id": selection.get("id"),
            "composicion": data,
            "resistencia_mpa": round(mpa, 3), "resistencia_kgcm2": round(kgcm2, 2),
            "clasificacion": clasificar(kgcm2), "co2_kg_m3": co2,
            "costo_sol_m3": cost, "relacion_ac": round(ac, 3),
        })
    reference_index = req.referencia if req.referencia is not None else 0
    reference = out[reference_index]
    for item in out:
        item["es_referencia"] = item["id"] == reference["id"]
        item["diferencias_referencia"] = {
            "resistencia_mpa": round(item["resistencia_mpa"] - reference["resistencia_mpa"], 3),
            "resistencia_kgcm2": round(item["resistencia_kgcm2"] - reference["resistencia_kgcm2"], 2),
            "costo_sol_m3": round(item["costo_sol_m3"] - reference["costo_sol_m3"], 2),
            "co2_kg_m3": round(item["co2_kg_m3"] - reference["co2_kg_m3"], 2),
            "relacion_ac": round(item["relacion_ac"] - reference["relacion_ac"], 3),
        }
        delta = item["diferencias_referencia"]
        item["explicacion_referencia"] = (
            "Es la referencia."
            if item["es_referencia"] else
            f"Respecto de {reference['nombre']}: resistencia "
            f"{delta['resistencia_kgcm2']:+.2f} kg/cm², costo "
            f"{delta['costo_sol_m3']:+.2f} S/. por m³, CO2 "
            f"{delta['co2_kg_m3']:+.2f} kg por m³ y relación a/c "
            f"{delta['relacion_ac']:+.3f}."
        )
    comparison = {
        "contexto_comun": context,
        "referencia_id": reference["id"],
        "referencia_nombre": reference["nombre"],
        "resultados": out,
        "explicacion": (
            "Las diferencias se calculan contra la referencia usando el mismo modelo, "
            "perfil ambiental, asentamiento, edad y parámetros del contexto común."
        ),
    }
    if req.guardar_comparacion:
        comparison_id = store_comparison(req.nombre or "Comparación", comparison, "Comparación de 2 a 20 composiciones")
        comparison["comparacion_id"] = comparison_id
    return comparison


@app.get("/comparaciones/{comparison_id}", tags=["Comparación"])
def obtener_comparacion(comparison_id: str):
    conn = get_db_connection()
    try:
        row = conn.execute("SELECT * FROM comparisons WHERE id = ?", (comparison_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Comparación no encontrada")
        payload = dict(row)
        payload["comparacion"] = json.loads(payload.pop("comparison_json"))
        return payload
    finally:
        conn.close()


def _pdf_text(value: Any) -> str:
    return str(value if value not in (None, "") else "No especificado").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


@app.post("/informes/comparacion.pdf", tags=["Informes"])
def informe_comparacion_pdf(req: CompararIn):
    comparison = comparar(req)
    resultados = comparison["resultados"]
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=14 * mm,
        leftMargin=14 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="ALINA - Informe técnico de comparación",
    )

    def number(value: Any, decimals: int = 2) -> str:
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return "-"

    best_resistance = max(resultados, key=lambda item: float(item["resistencia_kgcm2"]))
    lowest_cost = min(resultados, key=lambda item: float(item["costo_sol_m3"]))
    lowest_co2 = min(resultados, key=lambda item: float(item["co2_kg_m3"]))

    contexto = comparison["contexto_comun"] or {}
    exposure_names = {
        "normal": "Sin exposición especial",
        "sulfatos": "Sulfatos altos",
        "cloruros": "Cloruros o ambiente marino",
        "acidos": "Medio ácido",
    }
    context_exposure = exposure_names.get(str(contexto.get("exposicion_ambiental", "normal")), "Sin exposición especial")
    context_age = contexto.get("age", 28)
    context_slump = contexto.get("asentamiento_objetivo_mm", 100)

    story = [
        Paragraph("ALINA", styles["Title"]),
        Paragraph("Asistente para mezclas de concreto", styles["Heading2"]),
        Paragraph("Informe técnico de comparación", styles["Heading1"]),
        Spacer(1, 4 * mm),
        Table([
            ["Fecha", datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S") + " (hora local)"],
            ["Referencia", _pdf_text(comparison["referencia_nombre"])],
            ["Contexto", f"Exposición: {context_exposure}; Edad: {context_age} días; Asentamiento: {context_slump} mm"],
            ["Modelo activo", selected_model_name],
        ], colWidths=[38 * mm, 134 * mm], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF1F8")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])),
        Spacer(1, 6 * mm),
        Paragraph("Resumen ejecutivo", styles["Heading2"]),
    ]

    summary_rows = [["Criterio", "Mezcla destacada", "Resultado"]]
    summary_rows.extend([
        ["Mayor resistencia", best_resistance["nombre"], f"{number(best_resistance['resistencia_kgcm2'])} kg/cm²"],
        ["Menor costo", lowest_cost["nombre"], f"S/. {number(lowest_cost['costo_sol_m3'])} por m³"],
        ["Menor CO2", lowest_co2["nombre"], f"{number(lowest_co2['co2_kg_m3'])} kg/m³"],
    ])
    summary_table = Table(summary_rows, colWidths=[48 * mm, 72 * mm, 52 * mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B4F9C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F8FA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([summary_table, Spacer(1, 6 * mm), Paragraph("Comparación técnica", styles["Heading2"])])

    rows = [["Mezcla", "f'c kg/cm²", "MPa", "Costo S/./m³", "CO2 kg/m³", "Cemento kg/m³", "Δ f'c", "Clasificación"]]
    for item in resultados:
        composition = item["composicion"]
        delta = item["diferencias_referencia"]
        rows.append([
            _pdf_text(item["nombre"]),
            number(item["resistencia_kgcm2"]),
            number(item["resistencia_mpa"]),
            number(item["costo_sol_m3"]),
            number(item["co2_kg_m3"]),
            number(composition.get("cement"), 2),
            "Base" if item["es_referencia"] else f"{delta['resistencia_kgcm2']:+.2f}",
            _pdf_text(item["clasificacion"]),
        ])

    table = Table(rows, repeatRows=1, colWidths=[24 * mm, 20 * mm, 16 * mm, 24 * mm, 20 * mm, 24 * mm, 16 * mm, 28 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B4F9C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
        ("FONTSIZE", (0, 0), (-1, -1), 7.2),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F8FA")),
        ("ALIGN", (1, 1), (-2, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.extend([table, Spacer(1, 6 * mm), Paragraph("Análisis frente a la referencia", styles["Heading2"])])

    analysis_rows = [["Mezcla", "Interpretación frente a la referencia"]]
    for item in resultados:
        if item["es_referencia"]:
            interpretation = "Mezcla de referencia para esta comparación."
        else:
            interpretation = item["explicacion_referencia"].replace("CO₂", "CO2")
        analysis_rows.append([_pdf_text(item["nombre"]), _pdf_text(interpretation)])

    analysis_table = Table(analysis_rows, colWidths=[34 * mm, 138 * mm], repeatRows=1)
    analysis_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B4F9C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 1), (0, -1), colors.HexColor("#EAF1F8")),
        ("BACKGROUND", (1, 1), (1, -1), colors.HexColor("#F7F8FA")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]))
    story.append(analysis_table)

    story.extend([
        Spacer(1, 4 * mm),
        Table([["Alcance", "Los resultados son estimaciones estadísticas. Deben validarse mediante ensayos de laboratorio y aprobación del responsable técnico."]], colWidths=[32 * mm, 140 * mm], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F59E0B")),
            ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF7E6")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])),
    ])

    document.build(story)
    buffer.seek(0)
    filename = f"alina-informe-comparacion-{datetime.now().strftime('%Y%m%d%H%M%S')}.pdf"
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.post("/informes/evaluacion.pdf", tags=["Informes"])
def informe_evaluacion_pdf(req: EvaluationReportIn):
    composition = req.composition.model_dump()
    result = req.result
    buffer = BytesIO()
    styles = getSampleStyleSheet()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=16 * mm,
        leftMargin=16 * mm,
        topMargin=14 * mm,
        bottomMargin=14 * mm,
        title="ALINA - Informe técnico de evaluación",
    )

    def number(value: Any, decimals: int = 2) -> str:
        try:
            return f"{float(value):.{decimals}f}"
        except (TypeError, ValueError):
            return "-"

    resist_mpa = number(result.get("resistencia_mpa"), 2)
    resist_kgcm2 = number(result.get("resistencia_kgcm2"), 2)
    costo = number(result.get("costo_sol_m3"), 2)
    co2 = number(result.get("co2_kg_m3"), 2)
    relacion_ac = number(result.get("relacion_ac"), 3)
    clase = _pdf_text(result.get("clasificacion", "Por validar"))

    story = [
        Paragraph("ALINA", styles["Title"]),
        Paragraph("Asistente para mezclas de concreto", styles["Heading2"]),
        Paragraph("Informe técnico preliminar de evaluación", styles["Heading1"]),
        Spacer(1, 4 * mm),
        Table([
            ["Proyecto", _pdf_text(req.proyecto)],
            ["Responsable técnico", _pdf_text(req.responsable)],
            ["Fecha", req.fecha_generacion or datetime.now().astimezone().strftime("%d/%m/%Y %H:%M:%S") + " (hora local)"],
            ["Modelo activo", selected_model_name],
        ], colWidths=[42 * mm, 128 * mm], repeatRows=0, style=TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EAF1F8")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ])),
        Spacer(1, 6 * mm),
        Paragraph("Resumen de resultados", styles["Heading2"]),
    ]

    summary_rows = [["Indicador", "Resultado", "Unidad"]]
    summary_rows.extend([
        ["Resistencia estimada", resist_kgcm2, "kg/cm²"],
        ["Resistencia estimada", resist_mpa, "MPa"],
        ["Costo estimado", costo, "S/ por m³"],
        ["Huella de CO2", co2, "kg por m³"],
        ["Relación agua/cementante", relacion_ac, "a/c"],
        ["Clasificación", clase, "—"],
    ])
    summary_table = Table(summary_rows, colWidths=[68 * mm, 48 * mm, 54 * mm], repeatRows=1)
    summary_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B4F9C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F8FA")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([summary_table, Spacer(1, 6 * mm), Paragraph("Dosificación utilizada", styles["Heading2"])])

    dosage_rows = [["Material", "Cantidad", "Unidad"]]
    dosage_rows.extend([
        ["Cemento", number(composition.get("cement"), 2), "kg/m³"],
        ["Escoria BFS", number(composition.get("slag"), 2), "kg/m³"],
        ["Ceniza volante", number(composition.get("flyash"), 2), "kg/m³"],
        ["Agua", number(composition.get("water"), 2), "L/m³"],
        ["Superplastificante", number(composition.get("superplasticizer"), 2), "kg/m³"],
        ["Agregado grueso", number(composition.get("coarseaggregate"), 2), "kg/m³"],
        ["Agregado fino", number(composition.get("fineaggregate"), 2), "kg/m³"],
        ["Edad de ensayo", number(composition.get("age"), 0), "días"],
        ["Asentamiento objetivo", number(composition.get("asentamiento_objetivo_mm"), 1), "mm"],
    ])
    dosage_table = Table(dosage_rows, colWidths=[68 * mm, 48 * mm, 54 * mm], repeatRows=1)
    dosage_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B4F9C")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F7F8FA")),
        ("ALIGN", (1, 1), (1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 7),
        ("RIGHTPADDING", (0, 0), (-1, -1), 7),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.extend([dosage_table, Spacer(1, 6 * mm), Paragraph("Observaciones técnicas", styles["Heading2"])])

    durabilidad = result.get("durabilidad") or {}
    trabajabilidad = result.get("trabajabilidad") or {}
    observations = [
        f"Exposición seleccionada: {_pdf_text(durabilidad.get('nombre', 'Sin exposición especial'))}.",
        f"Recomendación: {_pdf_text(durabilidad.get('recomendacion', 'Validar los parámetros con el responsable del diseño.'))}",
        f"Trabajabilidad: {_pdf_text(trabajabilidad.get('control_requerido', 'Verificar asentamiento mediante ensayo de cono de Abrams.'))}",
        "Resultado preliminar: confirmar resistencia, durabilidad y asentamiento mediante ensayos de laboratorio y aprobación profesional.",
    ]
    for observation in observations:
        story.append(Paragraph("• " + _pdf_text(observation), styles["BodyText"]))
        story.append(Spacer(1, 2 * mm))

    story.extend([
        Spacer(1, 6 * mm),
        Table([["Alcance", "La predicción estadística es una herramienta de apoyo. No reemplaza ensayos, especificaciones técnicas ni aprobación profesional."]], colWidths=[32 * mm, 138 * mm], style=TableStyle([
            ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#F59E0B")),
            ("TEXTCOLOR", (0, 0), (0, 0), colors.white),
            ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#FFF7E6")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD0DC")),
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ])),
    ])

    document.build(story)
    buffer.seek(0)
    filename = f"alina-informe-evaluacion-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.pdf"
    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f'attachment; filename="{filename}"'})
