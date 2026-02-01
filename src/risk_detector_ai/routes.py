import os
from flask import Blueprint, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
import pandas as pd
import numpy as np
import joblib
from werkzeug.utils import secure_filename

from .features import build_feature_table

bp = Blueprint('main', __name__)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads', 'test_files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ML_DIR = os.path.join(os.path.dirname(__file__), 'ml_models')
RISK_BEHAVIOR_PATH = os.path.join(ML_DIR, 'risk_behavior_model.pkl')
ALLOWED_EXT = {'.csv'}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


@bp.route('/')
def index():
    data_path = os.path.join(os.path.dirname(__file__), 'data_treino', 'dados_juntos.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
        total_players = df['user_id'].nunique() if 'user_id' in df.columns else 0
        total_bets = df.shape[0]
    else:
        total_players = 0
        total_bets = 0
    return render_template('base.html', total_players=total_players, total_bets=total_bets)


@bp.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            df = pd.read_csv(save_path)
            df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
            required = {'user_id'}
            if not required.issubset(set(df.columns)):
                flash('CSV missing required columns (user_id)')
                return redirect(request.url)

            Xagg = build_feature_table(df)
            results = Xagg[['user_id']].copy()

            if os.path.exists(RISK_BEHAVIOR_PATH) and Xagg.shape[0] > 0:
                artifact = joblib.load(RISK_BEHAVIOR_PATH)
                preproc = artifact['preprocessor']
                model = artifact['model']
                feat_list = artifact['features']

                feat = [c for c in feat_list if c in Xagg.columns]
                X = Xagg[feat].fillna(0)
                X_trans = preproc.transform(X)
                raw_scores = model.score_samples(X_trans)
                minv = float(raw_scores.min())
                maxv = float(raw_scores.max())
                if maxv - minv > 1e-9:
                    norm = (raw_scores - minv) / (maxv - minv)
                else:
                    norm = np.zeros_like(raw_scores)
                results['risk_score'] = norm
            else:
                results['risk_score'] = 0.0

            out_name = f"results_{os.path.splitext(filename)[0]}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
            out_path = os.path.join(UPLOAD_FOLDER, out_name)
            results.to_csv(out_path, index=False)
            return redirect(url_for('main.results', filename=out_name))
    return render_template('upload.html')


@bp.route('/results')
def results():
    filename = request.args.get('filename')
    if not filename:
        files = sorted([f for f in os.listdir(UPLOAD_FOLDER) if f.startswith('results_')], reverse=True)
        return render_template('results_list.html', files=files)

    file_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        flash('Result file not found')
        return redirect(url_for('main.upload'))

    df = pd.read_csv(file_path)
    rows = df.to_dict(orient='records')
    cols = list(df.columns)

    if 'risk_score' in df.columns:
        counts, edges = np.histogram(df['risk_score'].fillna(0).values, bins=20)
        hist_labels = [round(float(e), 3) for e in edges[:-1]]
        hist_counts = counts.tolist()
    else:
        hist_labels = []
        hist_counts = []

    return render_template('results.html', rows=rows, cols=cols, filename=filename, hist_labels=hist_labels, hist_counts=hist_counts)


@bp.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)
