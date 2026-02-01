import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path

# Get the directory of this file
BASEDIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASEDIR, 'data', 'data_treino', 'dados_completos.csv')
MODEL_DIR = os.path.join(BASEDIR, 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)

# Import feature builder
from .features import build_feature_table


def train_isolation_forest(save_path=MODEL_DIR):
    """
    Train Isolation Forest on risk players dataset (positive class only).
    Learns the risk feature space to flag similar players.
    """
    print(f"Loading data from {DATA_PATH}...")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"Loaded {df.shape[0]} records with {df.shape[1]} columns")
    
    # Build aggregated features per user
    user_agg = build_feature_table(df)
    print(f"Built features for {user_agg.shape[0]} users")
    
    # Drop user_id and select features
    if 'user_id' in user_agg.columns:
        features = [c for c in user_agg.columns if c != 'user_id']
    else:
        features = list(user_agg.columns)
    
    X = user_agg[features].copy()
    
    # Identify numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]
    
    print(f"Numeric features: {numeric_cols}")
    print(f"Categorical features: {cat_cols}")
    
    # Build preprocessing pipeline
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    
    # Encode categorical features
    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('label_encoder', LabelEncoder())
    ]) if cat_cols else None
    
    # Build column transformer
    transformers = [('num', numeric_transformer, numeric_cols)]
    if cat_transformer and cat_cols:
        transformers.append(('cat', cat_transformer, cat_cols))
    
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'
    )
    
    # Fit preprocessor and transform data
    print("Preprocessing features...")
    X_prep = preprocessor.fit_transform(X)
    
    # Train Isolation Forest
    print("Training Isolation Forest...")
    iso = IsolationForest(
        n_estimators=300,
        contamination=0.01,  # Reduced from 0.05 (5%) to 0.01 (1%) for more assertive detection
        random_state=42,
        max_samples='auto',
        max_features=1.0,
        n_jobs=-1
    )
    iso.fit(X_prep)
    
    # Save artifact
    artifact = {
        'preprocessor': preprocessor,
        'model': iso,
        'features': features,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols
    }
    
    model_file = os.path.join(save_path, 'risk_model.pkl')
    joblib.dump(artifact, model_file)
    print(f"✅ Model saved to {model_file}")
    
    return artifact


if __name__ == '__main__':
    train_isolation_forest()
