@echo off
REM Quick Start - Risk Detector Auto-Healing ML Model
REM Este script inicia o app com o modelo treinado automaticamente se necessário

echo.
echo ============================================================
echo RISK DETECTOR - QUICK START
echo ============================================================
echo.

cd /d %~dp0

REM Verificar se está no diretório correto
if not exist "src\risk_detector_ai\app.py" (
    echo ERRO: Execute este script do diretório raiz do projeto
    echo Esperado: risk-detector\
    pause
    exit /b 1
)

echo [1/4] Verificando ambiente Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERRO: Python nao encontrado. Instale Python 3.10+
    pause
    exit /b 1
)
echo [1/4] Python OK

echo [2/4] Verificando dados de treinamento...
if not exist "src\risk_detector_ai\data\data_treino\dados_completos.csv" (
    echo ERRO: dados_completos.csv nao encontrado
    echo Procure: src\risk_detector_ai\data\data_treino\dados_completos.csv
    pause
    exit /b 1
)
echo [2/4] Dados OK

echo [3/4] Verificando model_trainer.py...
if not exist "src\risk_detector_ai\model_trainer.py" (
    echo ERRO: model_trainer.py nao encontrado
    pause
    exit /b 1
)
echo [3/4] Model trainer OK

echo [4/4] Iniciando app...
echo.
echo ============================================================
echo INICIANDO RISK DETECTOR COM AUTO-HEALING
echo ============================================================
echo.
echo [INFO] Se o modelo nao existir, sera treinado automaticamente
echo [INFO] App disponivel em: http://localhost:5000
echo [INFO] Upload em: http://localhost:5000/upload
echo.
echo Press CTRL+C para parar
echo.

REM Iniciar o app
python run.py

pause
