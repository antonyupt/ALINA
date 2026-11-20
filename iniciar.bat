@echo off
title ConcretoIA — Servidor Backend

echo.
echo  ============================================
echo   ConcretoIA - Cemento Yura S.A.
echo   Innova Sur 2026 - Reto 2
echo  ============================================
echo.

REM Ir a la raiz del proyecto (donde esta este .bat)
cd /d "%~dp0"

REM Verificar que existe el modelo
if not exist "modelo\model_selected.pkl" (
    echo [1/2] Entrenando modelos...
    python modelo\train_model.py
    if errorlevel 1 (
        echo ERROR: Fallo el entrenamiento. Verifica que esten instaladas las dependencias.
        echo Ejecuta: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo.
)

echo [2/2] Iniciando servidor FastAPI...
echo.
echo  API:   http://localhost:8000
echo  Docs:  http://localhost:8000/docs
echo  App:   Abre frontend\index.html en tu navegador
echo.
echo  Presiona Ctrl+C para detener.
echo.

python -m uvicorn backend.main:app --reload --port 8000 --host 0.0.0.0

pause
