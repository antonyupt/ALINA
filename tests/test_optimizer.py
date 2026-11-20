"""Pruebas del optimizador, la relación a/mc y la calidad del dataset de entrenamiento."""
import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import MODEL_OK, app, relacion_agua_cementante

client = TestClient(app)
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "modelo", "concrete_data.csv")
needs_model = pytest.mark.skipif(not MODEL_OK, reason="modelo no disponible en el entorno")


def test_relacion_agua_cementante_usa_todo_el_cementante():
    assert relacion_agua_cementante(300, 100, 50, 180) == pytest.approx(180 / 450)


def test_dataset_de_entrenamiento_es_real_y_sin_piso_artificial():
    df = pd.read_csv(CSV)
    assert len(df) >= 1000
    assert (df["csMPa"] <= df["csMPa"].min() + 1e-6).mean() < 0.05  # antes: 52% en 2.0 MPa
    assert 30 < df[df["age"] == 28]["csMPa"].mean() < 45          # concreto real a 28 días


@needs_model
def test_optimizar_cumple_objetivo_y_restricciones_pedidas():
    body = {
        "resistencia_objetivo_kgcm2": 300, "agua": 180, "edad": 28, "max_cement": 400,
        "usar_escoria": True, "usar_ceniza": False, "prioridad": "costo",
    }
    r = client.post("/optimizar", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["factible"] is True and data["alternativas"]
    for alt in data["alternativas"]:
        assert alt["resistencia_kgcm2"] >= 300          # antes devolvía ~25 kg/cm²
        assert alt["cement"] <= 400                      # respeta max_cement
        assert alt["flyash"] == 0                        # respeta usar_ceniza=False
        assert alt["water"] == 180 and alt["age"] == 28  # respeta agua y edad
        assert alt["cumple_reglas"] is True
        assert alt["relacion_ac"] == pytest.approx(
            relacion_agua_cementante(alt["cement"], alt["slag"], alt["flyash"], alt["water"]), abs=1e-3
        )
    costos = [a["costo_sol_m3"] for a in data["alternativas"][:1]]
    assert costos[0] == min(a["costo_sol_m3"] for a in data["alternativas"])


@needs_model
def test_optimizar_es_reproducible_y_reporta_ahorro_con_adiciones():
    body = {"resistencia_objetivo_kgcm2": 350, "agua": 180, "usar_escoria": True, "usar_ceniza": True}
    a, b = client.post("/optimizar", json=body).json(), client.post("/optimizar", json=body).json()
    assert a["alternativas"] == b["alternativas"]
    assert a["referencia_convencional"] is not None
    con_ahorro = [x for x in a["alternativas"] if x["adicion_pct"] > 0 and x["reduccion_co2_pct"] > 0]
    assert con_ahorro, "debe haber alternativas con adiciones que reduzcan CO2 frente a la convencional"


@needs_model
def test_optimizar_objetivo_inalcanzable_avisa_en_vez_de_inventar():
    body = {"resistencia_objetivo_kgcm2": 700, "agua": 200, "max_cement": 300}
    r = client.post("/optimizar", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["factible"] is False and data["advertencias"]
    assert all(not a["cumple_objetivo"] for a in data["alternativas"])


@needs_model
def test_perfil_agresivo_eleva_el_objetivo_efectivo():
    r = client.post("/optimizar", json={"resistencia_objetivo_kgcm2": 210, "exposicion_ambiental": "sulfatos"})
    assert r.status_code == 200
    data = r.json()
    assert data["objetivo_resistencia_efectivo_kgcm2"] == 280
    assert all(a["relacion_ac"] <= 0.45 for a in data["alternativas"])


def test_optimizar_valida_entradas():
    assert client.post("/optimizar", json={"resistencia_objetivo_kgcm2": 50}).status_code == 422
    assert client.post("/optimizar", json={"resistencia_objetivo_kgcm2": 300, "n_muestras": 10}).status_code == 422
