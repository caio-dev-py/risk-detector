#!/usr/bin/env python
"""
Risk Detector - Sistema de Diagnóstico
Verifica saúde do sistema e identifica problemas
"""
import os
import sys
import pandas as pd
from pathlib import Path

BASEDIR = Path(__file__).parent
RISK_DETECTOR_DIR = BASEDIR / 'src' / 'risk_detector_ai'
ML_MODELS_DIR = RISK_DETECTOR_DIR / 'ml_models'
DATA_DIR = RISK_DETECTOR_DIR / 'data' / 'data_treino'

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def check_file(path, name, required=True):
    """Verifica se um arquivo existe"""
    path_str = str(path)
    exists = os.path.exists(path)
    status = "✓ ENCONTRADO" if exists else "✗ FALTANDO"
    symbol = "✓" if exists else ("✗" if required else "⚠")
    print(f"  {symbol} {name:<40} {status}")
    
    if exists and path_str.endswith('.csv'):
        try:
            df = pd.read_csv(path, nrows=5)
            print(f"    └─ Shape: {pd.read_csv(path).shape}")
            print(f"    └─ Columns: {', '.join(list(df.columns)[:5])}...")
        except Exception as e:
            print(f"    └─ ERRO ao ler: {e}")
    
    return exists

def check_directory(path, name):
    """Verifica se um diretório existe"""
    exists = os.path.isdir(path)
    status = "✓ EXISTE" if exists else "✗ NÃO EXISTE"
    symbol = "✓" if exists else "✗"
    print(f"  {symbol} {name:<40} {status}")
    return exists

def main():
    print("\n")
    print_header("RISK DETECTOR - SISTEMA DE DIAGNÓSTICO")
    
    # 1. Verificar estrutura de diretórios
    print_header("1. ESTRUTURA DE DIRETÓRIOS")
    check_directory(RISK_DETECTOR_DIR, "src/risk_detector_ai/")
    check_directory(ML_MODELS_DIR, "ml_models/")
    check_directory(DATA_DIR, "data/data_treino/")
    check_directory(RISK_DETECTOR_DIR / 'uploads', "uploads/")
    check_directory(RISK_DETECTOR_DIR / 'templates', "templates/")
    check_directory(RISK_DETECTOR_DIR / 'static', "static/")
    
    # 2. Verificar arquivos essenciais
    print_header("2. ARQUIVOS ESSENCIAIS")
    
    essential_files = [
        (RISK_DETECTOR_DIR / 'app.py', 'app.py', True),
        (RISK_DETECTOR_DIR / 'model_trainer.py', 'model_trainer.py', True),
        (RISK_DETECTOR_DIR / 'features.py', 'features.py', True),
        (RISK_DETECTOR_DIR / 'routes.py', 'routes.py', False),
        (RISK_DETECTOR_DIR / '__init__.py', '__init__.py', True),
    ]
    
    for path, name, required in essential_files:
        check_file(path, name, required)
    
    # 3. Verificar dados
    print_header("3. DADOS DE TREINAMENTO")
    data_path = DATA_DIR / 'dados_completos.csv'
    if check_file(data_path, 'dados_completos.csv', True):
        try:
            df = pd.read_csv(data_path)
            print(f"    └─ Registros: {len(df):,}")
            print(f"    └─ Colunas: {len(df.columns)}")
            if 'user_id' in df.columns:
                n_players = df['user_id'].nunique()
                print(f"    └─ Jogadores únicos: {n_players}")
        except Exception as e:
            print(f"    └─ ERRO: {e}")
    
    # 4. Verificar modelo
    print_header("4. MODELO DE MACHINE LEARNING")
    model_path = ML_MODELS_DIR / 'risk_model.pkl'
    compat_path = ML_MODELS_DIR / 'risk_behavior_model.pkl'
    
    model_exists = check_file(model_path, 'risk_model.pkl', False)
    compat_exists = check_file(compat_path, 'risk_behavior_model.pkl', False)
    
    if model_exists or compat_exists:
        print(f"    └─ STATUS: MODELO DISPONÍVEL ✓")
    else:
        print(f"    └─ STATUS: MODELO FALTANDO (será treinado no startup)")
    
    # 5. Verificar dependências Python
    print_header("5. DEPENDÊNCIAS PYTHON")
    
    dependencies = [
        'flask',
        'pandas',
        'numpy',
        'scikit-learn',
        'joblib',
    ]
    
    for pkg in dependencies:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg:<40} INSTALADO")
        except ImportError:
            print(f"  ✗ {pkg:<40} FALTANDO")
    
    # 6. Resumo e recomendações
    print_header("6. RESUMO & RECOMENDAÇÕES")
    
    all_ok = (
        RISK_DETECTOR_DIR.exists() and
        (ML_MODELS_DIR / 'risk_model.pkl').exists() or (ML_MODELS_DIR / 'risk_behavior_model.pkl').exists() and
        data_path.exists()
    )
    
    if all_ok:
        print("  ✓ SISTEMA PRONTO PARA USAR")
        print()
        print("  Próximo passo:")
        print("    python run.py")
    else:
        print("  ⚠ SISTEMA PRECISA DE SETUP")
        print()
        print("  Próximos passos:")
        print("    1. python -m src.risk_detector_ai.model_trainer")
        print("    2. python run.py")
    
    print("\n" + "="*60 + "\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n✗ ERRO: {e}\n")
        sys.exit(1)
