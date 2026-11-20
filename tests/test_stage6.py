import sqlite3
import uuid

import pytest
from fastapi.testclient import TestClient

from backend.main import DB_PATH, MODEL_OK, app


client = TestClient(app)

MIXTURE = {
    "cement": 380,
    "slag": 0,
    "flyash": 0,
    "water": 185,
    "superplasticizer": 6,
    "coarseaggregate": 950,
    "fineaggregate": 750,
    "age": 28,
}


def test_persistent_sources_are_available_to_assistant():
    suffix = uuid.uuid4().hex
    search = client.post(
        "/busquedas",
        json={"query": f"resistencia {suffix}", "filters": {"edad": 28}, "result_count": 2},
    )
    assert search.status_code == 200
    evaluation = client.post(
        "/evaluaciones",
        json={
            "composition": MIXTURE,
            "result": {"resistencia_kgcm2": 321.5},
            "model_name": "test",
            "exposure_profile": "normal",
        },
    )
    assert evaluation.status_code == 200
    evaluation_id = evaluation.json()["id"]
    try:
        assistant = client.post("/asistente/consultar", json={"pregunta": f"resistencia {suffix}"})
        assert assistant.status_code == 200
        payload = assistant.json()
        assert payload["fuentes_consultadas"] == ["busquedas", "evaluaciones", "comparaciones"]
        assert any(source["tipo"] == "búsqueda" for source in payload["fuentes"])
        assert "321.50" in payload["respuesta"]
    finally:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM evaluations WHERE id = ?", (evaluation_id,))
            conn.execute("DELETE FROM search_history WHERE query = ?", (f"resistencia {suffix}",))
            conn.commit()


def test_assistant_validates_input_and_list_limits():
    assert client.post("/asistente/consultar", json={"pregunta": "x"}).status_code == 422
    assert client.get("/evaluaciones?limit=0").status_code == 422
    assert client.get("/busquedas?limit=201").status_code == 422
    assert client.get("/comparaciones?limit=0").status_code == 422


def test_comparison_rejects_invalid_selection_count():
    response = client.post("/comparar", json={"mezclas": [MIXTURE]})
    assert response.status_code == 422


def test_optimizer_rejects_invalid_material_ranges():
    response = client.post(
        "/optimizar",
        json={
            "resistencia_objetivo_kgcm2": 210,
            "material_ranges": {"cement": {"min": 500, "max": 200, "step": 25}},
        },
    )
    assert response.status_code == (422 if MODEL_OK else 503)


@pytest.mark.skipif(not MODEL_OK, reason="modelo no disponible en el entorno")
def test_comparison_is_consistent_for_same_reference():
    response = client.post("/comparar", json={"mezclas": [MIXTURE, dict(MIXTURE)]})
    assert response.status_code == 200
    results = response.json()["resultados"]
    assert results[0]["es_referencia"] is True
    assert results[1]["diferencias_referencia"]["resistencia_kgcm2"] == 0
    assert results[1]["diferencias_referencia"]["costo_sol_m3"] == 0


@pytest.mark.skipif(not MODEL_OK, reason="modelo no disponible en el entorno")
def test_pdf_endpoints_return_pdf_and_comparison_can_persist():
    payload = {"mezclas": [MIXTURE, dict(MIXTURE, cement=400)], "guardar_comparacion": True}
    comparison = client.post("/comparar", json=payload)
    assert comparison.status_code == 200
    comparison_id = comparison.json()["comparacion_id"]
    try:
        pdf = client.post("/informes/comparacion.pdf", json=payload)
        assert pdf.status_code == 200
        assert pdf.headers["content-type"] == "application/pdf"
        assert pdf.content.startswith(b"%PDF")
        evaluation_pdf = client.post(
            "/informes/evaluacion.pdf",
            json={"composition": MIXTURE, "result": {"resistencia_kgcm2": 210}},
        )
        assert evaluation_pdf.status_code == 200
        assert evaluation_pdf.content.startswith(b"%PDF")
        listing = client.get("/comparaciones?limit=1")
        assert listing.status_code == 200
        assert any(item["id"] == comparison_id for item in listing.json()["comparaciones"])
    finally:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DELETE FROM comparisons WHERE id = ?", (comparison_id,))
            conn.commit()
