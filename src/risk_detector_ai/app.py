"""
Risk Detector Flask Application with Auto-Healing Model Training
"""
import os
import io
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib

from .features import build_feature_table
from .model_trainer import ensure_model_exists, load_model, train_model

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASEDIR, 'ml_models')
INSTANCE_DIR = os.path.join(BASEDIR, 'instance')
UPLOAD_DIR = os.path.join(BASEDIR, 'uploads')
TEMPLATES_DIR = os.path.join(BASEDIR, 'templates')

# Model paths
MODEL_PATH = os.path.join(ML_DIR, 'risk_model.pkl')
COMPAT_MODEL_PATH = os.path.join(ML_DIR, 'risk_behavior_model.pkl')

# Training data path
DATA_PATH = os.path.join(BASEDIR, 'data', 'data_treino', 'dados_completos.csv')

# Create required directories
for dir_path in [ML_DIR, INSTANCE_DIR, UPLOAD_DIR, TEMPLATES_DIR]:
    os.makedirs(dir_path, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}


def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__, 
                template_folder=TEMPLATES_DIR, 
                static_folder=os.path.join(BASEDIR, 'static'))
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key')
    
    # AUTO-HEALING: Ensure model exists on startup
    print("\n" + "="*60)
    print("RISK DETECTOR - STARTUP")
    print("="*60)
    if not ensure_model_exists():
        print("⚠ WARNING: Model could not be trained. Analysis will not work properly.")
    print("="*60 + "\n")
    
    # ======================== ROUTES ========================
    
    @app.route('/')
    def index():
        """Home page dashboard"""
        try:
            data_path = os.path.join(BASEDIR, 'data', 'data_treino', 'dados_completos.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, low_memory=False)
                df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
                total_players = df['user_id'].nunique() if 'user_id' in df.columns else 0
                total_bets = df.shape[0]
            else:
                total_players = 0
                total_bets = 0
            
            return render_template('index.html', total_players=total_players, total_bets=total_bets)
        except Exception as e:
            print(f"Error loading dashboard: {e}")
            return render_template('index.html', total_players=0, total_bets=0)
    
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        """Upload CSV and run risk analysis"""
        if request.method == 'GET':
            return render_template('upload.html')
        
        # POST - handle file upload
        if 'file' not in request.files:
            flash('Arquivo não enviado', 'error')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('Nenhum arquivo selecionado', 'error')
            return redirect(request.url)
        
        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            flash('Apenas arquivos CSV são permitidos', 'error')
            return redirect(request.url)
        
        try:
            # Read uploaded CSV
            stream = io.StringIO(file.stream.read().decode('utf-8', errors='ignore'))
            df = pd.read_csv(stream, low_memory=False)
            
            print(f"\n[Upload] Processing file: {filename}")
            print(f"[Upload] Shape: {df.shape}")
            
            # Normalize columns
            df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
            
            # Build features (aggregates by player)
            print("[Upload] Building player profiles...")
            user_agg = build_feature_table(df)
            print(f"[Upload] Created {len(user_agg)} player profiles")
            
            # Load model
            model_artifact = load_model(MODEL_PATH)
            if model_artifact is None:
                # Try compatibility path
                model_artifact = load_model(COMPAT_MODEL_PATH)
            
            if model_artifact is None:
                flash('Modelo não disponível. Tentando treinar...', 'warning')
                try:
                    train_model()
                    model_artifact = load_model(MODEL_PATH)
                except Exception as train_error:
                    print(f"[Upload] Training failed: {train_error}")
                    flash(f'Erro ao treinar modelo: {train_error}', 'error')
                    return redirect(request.url)
            
            # Score players (simplified - no complex pipeline)
            scaler = model_artifact['scaler']
            model = model_artifact['model']
            features = model_artifact['features']
            
            # Select available features
            feat = [c for c in features if c in user_agg.columns]
            X = user_agg[feat].copy()
            X = X.fillna(X.median(numeric_only=True))
            
            print(f"[Upload] Scoring {len(X)} players with {len(feat)} features...")
            X_scaled = scaler.transform(X)
            
            # Get anomaly scores (higher = more anomalous = more risk)
            raw_scores = model.score_samples(X_scaled)
            
            # Normalize to 0-1 range
            minv = float(raw_scores.min())
            maxv = float(raw_scores.max())
            if maxv - minv > 1e-9:
                risk_scores = (raw_scores - minv) / (maxv - minv)
            else:
                risk_scores = np.zeros_like(raw_scores)
            
            # Build results
            results_df = user_agg[['user_id']].copy()
            results_df['risk_score'] = risk_scores
            
            # Count flagged players
            high_risk = (risk_scores >= 0.5).sum()
            print(f"[Upload] ✓ Analysis complete: {high_risk}/{len(risk_scores)} flagged as high risk")
            
            # Save results
            timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
            out_name = f"results_{os.path.splitext(filename)[0]}_{timestamp}.csv"
            out_path = os.path.join(UPLOAD_DIR, out_name)
            results_df.to_csv(out_path, index=False)
            
            flash(f'✅ Análise concluída! {high_risk}/{len(risk_scores)} jogadores identificados como risco.', 'success')
            return redirect(url_for('results', filename=out_name))
            
        except Exception as e:
            print(f"[Upload] Error: {e}")
            flash(f'Erro ao processar arquivo: {str(e)}', 'error')
            return redirect(request.url)
    
    
    @app.route('/results')
    def results():
        """Display analysis results"""
        filename = request.args.get('filename')
        
        if not filename:
            # Show list of results
            files = []
            try:
                for fn in os.listdir(UPLOAD_DIR):
                    if fn.startswith('results_') and fn.endswith('.csv'):
                        path = os.path.join(UPLOAD_DIR, fn)
                        stat = os.stat(path)
                        df = pd.read_csv(path)
                        
                        # Calculate stats
                        risk_scores = df['risk_score'].fillna(0)
                        high_risk_count = (risk_scores >= 0.5).sum()
                        total_players = len(df)
                        
                        files.append({
                            'name': fn,
                            'mtime': stat.st_mtime,
                            'mtime_str': pd.to_datetime(stat.st_mtime, unit='s').strftime('%d/%m/%Y %H:%M:%S'),
                            'size': stat.st_size,
                            'total_players': total_players,
                            'high_risk': high_risk_count,
                            'avg_risk': float(risk_scores.mean())
                        })
                files = sorted(files, key=lambda x: x['mtime'], reverse=True)
            except Exception as e:
                print(f"Error listing results: {e}")
            
            return render_template('results_list.html', files=files)
        
        # Load specific result file
        try:
            safe_filename = secure_filename(filename)
            file_path = os.path.join(UPLOAD_DIR, safe_filename)
            
            if not os.path.exists(file_path):
                flash('Arquivo de resultado não encontrado', 'error')
                return redirect(url_for('results'))
            
            df = pd.read_csv(file_path)
            
            # Sort by risk score descending
            df = df.sort_values('risk_score', ascending=False)
            rows = df.to_dict(orient='records')
            cols = list(df.columns)
            
            # Compute statistics
            risk_scores = df['risk_score'].fillna(0)
            total_players = len(df)
            high_risk = (risk_scores >= 0.5).sum()
            medium_risk = ((risk_scores >= 0.3) & (risk_scores < 0.5)).sum()
            low_risk = (risk_scores < 0.3).sum()
            avg_risk = float(risk_scores.mean())
            
            # Compute histogram
            counts, edges = np.histogram(risk_scores.values, bins=20)
            hist_labels = [round(float(e), 2) for e in edges[:-1]]
            hist_counts = counts.tolist()
            
            return render_template('results.html',
                                 rows=rows,
                                 cols=cols,
                                 filename=safe_filename,
                                 hist_labels=hist_labels,
                                 hist_counts=hist_counts,
                                 total_players=total_players,
                                 high_risk=high_risk,
                                 medium_risk=medium_risk,
                                 low_risk=low_risk,
                                 avg_risk=avg_risk)
        
        except Exception as e:
            print(f"Error loading results: {e}")
            flash(f'Erro ao carregar resultado: {str(e)}', 'error')
            return redirect(url_for('results'))
    
    
    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        """Download results CSV file"""
        try:
            safe_filename = secure_filename(filename)
            return send_from_directory(UPLOAD_DIR, safe_filename)
        except Exception as e:
            flash(f'Erro ao baixar arquivo: {str(e)}', 'error')
            return redirect(url_for('results'))
    
    
    return app


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', 
            port=int(os.environ.get('SERVER_PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG', 'False') == 'True')
