"""
Model Trainer with Auto-Healing
Automatically trains and saves the risk detection model if missing
"""
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from .features import build_feature_table

BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASEDIR, 'data', 'data_treino', 'dados_completos.csv')
MODEL_DIR = os.path.join(BASEDIR, 'ml_models')
MODEL_PATH = os.path.join(MODEL_DIR, 'risk_model.pkl')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

os.makedirs(MODEL_DIR, exist_ok=True)


def clean_currency_column(series):
    """Convert 'R$ x.xxx,xx' format to float"""
    if series.dtype == 'object':
        series = series.str.replace('R$', '', regex=False)
        series = series.str.replace('.', '', regex=False)
        series = series.str.replace(',', '.', regex=False)
    return pd.to_numeric(series, errors='coerce')


def preprocess_training_data(df):
    """
    CRITICAL: Preprocess data before training
    - Group by ID Jogador (user_id)
    - Clean currency columns
    - Create player profiles with aggregated features
    """
    df = df.copy()
    
    # Normalize column names
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    
    # Clean currency columns if they exist with 'R$' format
    currency_cols = [col for col in df.columns if 
                     any(indicator in df[col].astype(str).iloc[0] if len(df) > 0 else False 
                         for indicator in ['R$', 'r$'])]
    
    for col in currency_cols:
        df[col] = clean_currency_column(df[col])
    
    # Use build_feature_table to aggregate by player (CRITICAL)
    player_profiles = build_feature_table(df)
    
    return player_profiles


def train_model(data_path=DATA_PATH, save_path=MODEL_DIR):
    """
    Train Isolation Forest on player risk profiles
    Uses ONLY aggregated player data, not individual bets
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training data not found at {data_path}")
    
    print(f"[Model Training] Loading training data from {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"[Model Training] Preprocessing data - aggregating by player...")
    user_agg = preprocess_training_data(df)
    
    print(f"[Model Training] Created {len(user_agg)} player profiles")
    
    # Prepare features for training
    if 'user_id' in user_agg.columns:
        features = [c for c in user_agg.columns if c != 'user_id']
    else:
        features = list(user_agg.columns)
    
    X = user_agg[features].copy()
    
    # Fill NaN with median
    print(f"[Model Training] Filling missing values...")
    X = X.fillna(X.median(numeric_only=True))
    
    # Standardize features
    print(f"[Model Training] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train Isolation Forest
    print(f"[Model Training] Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    
    # Package model artifact (simplified - no complex pipeline)
    artifact = {
        'scaler': scaler,
        'model': iso,
        'features': features,
        'training_samples': len(user_agg)
    }
    
    # Save model and scaler
    model_output = os.path.join(save_path, 'risk_model.pkl')
    joblib.dump(artifact, model_output)
    print(f"[Model Training] ✓ Model saved to {model_output}")
    
    # Also save for backward compatibility
    compat_output = os.path.join(save_path, 'risk_behavior_model.pkl')
    joblib.dump(artifact, compat_output)
    print(f"[Model Training] ✓ Model saved (compat) to {compat_output}")
    
    return artifact


def load_model(model_path=MODEL_PATH):
    """Load trained model artifact"""
    if not os.path.exists(model_path):
        return None
    try:
        artifact = joblib.load(model_path)
        return artifact
    except Exception as e:
        print(f"[Model Loading] Error loading model: {e}")
        return None


def ensure_model_exists():
    """
    AUTO-HEALING: Check if model exists on startup
    If NOT, immediately trigger training using historical data
    """
    if os.path.exists(MODEL_PATH) and os.path.exists(os.path.join(MODEL_DIR, 'risk_behavior_model.pkl')):
        print("[Model] ✓ Risk detection model found")
        return True
    
    print("[Model] ⚠ Model not found - triggering auto-training...")
    
    if not os.path.exists(DATA_PATH):
        print(f"[Model] ✗ Training data not found at {DATA_PATH}")
        return False
    
    try:
        train_model()
        print("[Model] ✓ Auto-training completed successfully")
        return True
    except Exception as e:
        print(f"[Model] ✗ Auto-training failed: {e}")
        return False


if __name__ == '__main__':
    print("=" * 60)
    print("RISK DETECTOR - MODEL TRAINER")
    print("=" * 60)
    train_model()
    print("=" * 60)
    print("Training completed!")
