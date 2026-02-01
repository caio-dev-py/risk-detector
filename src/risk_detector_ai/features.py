import numpy as np
import pandas as pd


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # normalize column names
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]

    def coerce_numeric(series):
        s = series.astype(str).str.replace('"', '').str.replace('\u00a0', ' ').str.strip()
        s = s.str.replace(',', '', regex=False)
        return pd.to_numeric(s, errors='coerce')

    for col in ['stake_amount', 'amount', 'gain_amount', 'odds', 'feature_amount']:
        if col in df.columns:
            df[col] = coerce_numeric(df[col])

    for c in ['created_at', 'settled_at']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    group = df.groupby('user_id')
    agg = pd.DataFrame()
    agg['bet_count'] = group.size()

    # numeric-only aggregations
    if 'stake_amount' in df.columns:
        agg['total_stake'] = group['stake_amount'].sum(numeric_only=True)
        agg['avg_stake'] = group['stake_amount'].mean(numeric_only=True)
    if 'odds' in df.columns:
        agg['avg_odds'] = group['odds'].mean(numeric_only=True)
    if 'gain_amount' in df.columns and 'stake_amount' in df.columns:
        agg['return_ratio'] = (group['gain_amount'].sum(numeric_only=True) / (group['stake_amount'].sum(numeric_only=True) + 1e-9))

    if 'created_at' in df.columns:
        now = pd.Timestamp.now(tz=None)
        # safe apply
        last = group['created_at'].max()
        agg['days_since_last_bet'] = last.apply(lambda dt: (now - pd.to_datetime(dt)).days if pd.notnull(dt) else np.nan)

    agg = agg.replace([np.inf, -np.inf], np.nan)
    return agg.reset_index()
