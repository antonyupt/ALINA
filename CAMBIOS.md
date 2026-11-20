# Cambios en ConcretoIA

1. **Dataset real.** `modelo/concrete_data.csv` era sintético (52 % de filas fijadas en 2 MPa). Ahora es el
   dataset de Yeh (UCI, 1030 ensayos). `train_model.py` rechaza datos sintéticos y quita 25 réplicas exactas.
   Resultado: XGBoost R² 0.937, MAE 2.7 MPa (CV 0.935).
2. **Optimizador reescrito** (`POST /optimizar`). Antes devolvía mezclas de ~25 kg/cm² e ignoraba agua, edad,
   cemento máximo y escoria/ceniza. Ahora muestrea 30 000 mezclas, las predice en lote, exige objetivo + margen
   (MAE), reglas y durabilidad, y devuelve alternativas distintas con % de adiciones y ahorro de CO₂/costo frente
   a la mejor mezcla sin adiciones. Si el objetivo es inalcanzable lo avisa.
3. **a/mc estándar** `agua/(cemento+escoria+ceniza)` en backend y frontend (antes ponderaba 0.4 y 0.3).
4. **Rangos** de escoria (250) y ceniza (150) ampliados para explorar cementos altamente adicionados.
5. **Base de datos:** `model_versions`/`data_versions` ya no se duplican en cada arranque.
6. **Frontend:** muestra % de adiciones, reducción de CO₂ y advertencias del optimizador.
7. **Tests:** `tests/test_optimizer.py` (7 nuevos). Total 13.

Primer uso: `instalar.bat` (instala dependencias y entrena; los .pkl no se incluyen).
