import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

BASEDIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASEDIR, 'data', 'data_treino', 'dados_completos.csv')
MODEL_DIR = os.path.join(BASEDIR, 'ml_models')
os.makedirs(MODEL_DIR, exist_ok=True)


from .features import build_feature_table


def train_oneclass(save_path=MODEL_DIR):
    df = pd.read_csv(DATA_PATH)
    user_agg = build_feature_table(df)

    # drop user_id for training
    if 'user_id' in user_agg.columns:
        features = [c for c in user_agg.columns if c != 'user_id']
    else:
        features = list(user_agg.columns)

    X = user_agg[features].copy()

    # Identify numeric and categorical
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    # Build OneHotEncoder in a sklearn-version compatible way
    try:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        # newer sklearn uses 'sparse_output' instead of 'sparse'
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    cat_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('ohe', ohe)
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', cat_transformer, cat_cols)
        ],
        remainder='drop'
    )

    # Fit preprocessor
    X_prep = preprocessor.fit_transform(X)

    # Train IsolationForest on positive (risk) examples
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_prep)

    artifact = {
        'preprocessor': preprocessor,
        'model': iso,
        'features': features,
        'numeric_cols': numeric_cols,
        'cat_cols': cat_cols
    }

    out_file = os.path.join(save_path, 'risk_behavior_model.pkl')
    joblib.dump(artifact, out_file)
    print(f"Saved one-class artifact to {out_file}")


if __name__ == '__main__':
    train_oneclass()
