@echo off
title ConcretoIA — Instalacion de dependencias

echo.
echo  ============================================
echo   ConcretoIA - Instalacion de dependencias
echo  ============================================
echo.

cd /d "%~dp0"

echo Instalando dependencias Python...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Fallo la instalacion. Verifica que Python este en tu PATH.
    pause
    exit /b 1
)

echo.
echo Entrenando modelos...
python modelo\train_model.py

echo.
echo  ============================================
echo   Instalacion completada.
echo   Ejecuta iniciar.bat para correr el sistema.
echo  ============================================
echo.
pause
