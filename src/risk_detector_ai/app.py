import os
import io
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

BASEDIR = os.path.dirname(os.path.abspath(__file__))
ML_DIR = os.path.join(BASEDIR, 'ml_models')
INSTANCE_DIR = os.path.join(BASEDIR, 'instance')
UPLOAD_DIR = os.path.join(BASEDIR, 'uploads')
TEMPLATES_DIR = os.path.join(BASEDIR, 'templates')
MODEL_PATH = os.path.join(ML_DIR, 'risk_model.pkl')
TRAINING_CSV = os.path.join(BASEDIR, 'dados_completos.csv')

os.makedirs(ML_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'csv'}


def create_app():
    app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=os.path.join(BASEDIR, 'static'))
    app.config['UPLOAD_FOLDER'] = UPLOAD_DIR
    app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret')

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        if request.method == 'GET':
            return render_template('index.html')

        # POST - handle uploaded CSV
        if 'file' not in request.files:
            flash('Arquivo não enviado')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('Nenhum arquivo selecionado')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        if not allowed_file(filename):
            flash('Apenas CSVs são permitidos')
            return redirect(request.url)

        # read uploaded csv into pandas
        stream = io.StringIO(f.stream.read().decode('utf-8', errors='ignore'))
        df = pd.read_csv(stream, low_memory=False)

        # process and aggregate
        agg = aggregate_by_player(df)

        # load model
        if not os.path.exists(MODEL_PATH):
            # try to train if missing
            train_model()
        artifact = joblib.load(MODEL_PATH)
        preprocessor = artifact['preprocessor']
        model = artifact['model']
        features = artifact['features']

        X = agg[features].copy()
        X_prep = preprocessor.transform(X)

        raw_scores = model.decision_function(X_prep)  # higher = more normal
        # convert to risk score (1 means anomaly/high risk)
        # invert and normalize
        minv = float(np.min(raw_scores))
        maxv = float(np.max(raw_scores))
        if maxv - minv == 0:
            norm = np.zeros_like(raw_scores)
        else:
            norm = (raw_scores - minv) / (maxv - minv)
        risk = 1.0 - norm

        agg = agg.reset_index(drop=True)
        agg['risk_score'] = risk

        # compute rule-based reasons using population statistics
        reasons_map = explain_risk_reasons(agg)

        # build results list including reasons
        results = []
        for _, row in agg.iterrows():
            uid = row.get('id_jogador') if 'id_jogador' in row.index else row.get('user_id', None)
            res = {
                'id': uid,
                'risk_score': float(row['risk_score']),
                'avg_stake': float(row.get('avg_stake', np.nan)),
                'win_rate': float(row.get('return_ratio', np.nan)),
                'reasons': reasons_map.get(uid, ['Padrão incomum detectado'])
            }
            results.append(res)

        # sort by risk desc
        results = sorted(results, key=lambda r: r['risk_score'], reverse=True)

        # save uploaded file for audit
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.stream.seek(0)
        f.save(save_path)

        return render_template('index.html', results=results, total=len(results))

    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

    @app.route('/analyses')
    def list_analyses():
        files = []
        for fn in os.listdir(app.config['UPLOAD_FOLDER']):
            if fn.lower().endswith('.csv'):
                path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
                stat = os.stat(path)
                files.append({'name': fn, 'mtime': stat.st_mtime})
        files = sorted(files, key=lambda x: x['mtime'], reverse=True)
        # format mtime
        for f in files:
            f['mtime_str'] = pd.to_datetime(f['mtime'], unit='s').strftime('%Y-%m-%d %H:%M:%S')
        return render_template('analyses.html', files=files)

    @app.route('/view')
    def view_analysis():
        filename = request.args.get('filename')
        if not filename:
            flash('Arquivo não informado')
            return redirect(url_for('list_analyses'))
        safe = secure_filename(filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], safe)
        if not os.path.exists(path):
            flash('Arquivo não encontrado')
            return redirect(url_for('list_analyses'))

        df = pd.read_csv(path, low_memory=False)
        agg = aggregate_by_player(df)

        if not os.path.exists(MODEL_PATH):
            train_model()
        artifact = joblib.load(MODEL_PATH)
        preprocessor = artifact['preprocessor']
        model = artifact['model']
        features = artifact['features']

        X = agg[features].copy()
        X_prep = preprocessor.transform(X)
        raw_scores = model.decision_function(X_prep)
        minv = float(np.min(raw_scores))
        maxv = float(np.max(raw_scores))
        norm = (raw_scores - minv) / (maxv - minv) if maxv - minv != 0 else np.zeros_like(raw_scores)
        risk = 1.0 - norm
        agg = agg.reset_index(drop=True)
        agg['risk_score'] = risk
        reasons_map = explain_risk_reasons(agg)

        results = []
        for _, row in agg.iterrows():
            uid = row.get('id_jogador')
            results.append({
                'id': uid,
                'risk_score': float(row['risk_score']),
                'avg_stake': float(row.get('avg_stake', np.nan)),
                'win_rate': float(row.get('return_ratio', np.nan)),
                'reasons': reasons_map.get(uid, ['Padrão incomum detectado'])
            })
        results = sorted(results, key=lambda r: r['risk_score'], reverse=True)
        return render_template('analysis_view.html', filename=safe, results=results, total=len(results))

    return app


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def clean_currency_series(s: pd.Series) -> pd.Series:
    if s.dtype == object:
        # remove currency symbol, thousand sep '.' and replace decimal comma
        s2 = s.astype(str).str.replace(r'[^0-9,.-]', '', regex=True).str.replace('.', '', regex=False).str.replace(',', '.', regex=False)
        return pd.to_numeric(s2, errors='coerce')
    return pd.to_numeric(s, errors='coerce')


def aggregate_by_player(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # normalize columns
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    # map candidate id column names
    id_col = None
    for candidate in ['id_jogador', 'user_id', 'id']:
        if candidate in df.columns:
            id_col = candidate
            break
    if id_col is None:
        # try original name
        if 'id jogador' in df.columns:
            id_col = 'id_jogador'
    if id_col is None:
        raise ValueError('Coluna de id do jogador não encontrada (procure por ID_JOGADOR ou user_id)')

    # currency columns (Portuguese names per spec)
    currency_cols = ['total_depósitos_(r$)', 'net_deposits_(r$)', 'volume_sportsbook_(r$)']
    for col in currency_cols:
        if col in df.columns:
            df[col] = clean_currency_series(df[col])

    # numeric conversions for common columns
    for col in ['stake_amount', 'amount', 'gain_amount', 'odds', 'feature_amount', 'valor']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # ensure datetime conversion for created_at if exists
    if 'created_at' in df.columns:
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

    group = df.groupby(id_col, dropna=False)
    agg = pd.DataFrame()
    agg['bet_count'] = group.size()

    if 'stake_amount' in df.columns:
        agg['total_stake'] = group['stake_amount'].sum(numeric_only=True)
        agg['avg_stake'] = group['stake_amount'].mean(numeric_only=True)
    if 'odds' in df.columns:
        agg['avg_odds'] = group['odds'].mean(numeric_only=True)
    if 'gain_amount' in df.columns and 'stake_amount' in df.columns:
        agg['return_ratio'] = (group['gain_amount'].sum(numeric_only=True) / (group['stake_amount'].sum(numeric_only=True) + 1e-9))
    # max deposit if exists
    for dep in ['total_depósitos_(r$)', 'net_deposits_(r$)', 'volume_sportsbook_(r$)']:
        if dep in df.columns:
            agg['max_deposit'] = group[dep].max(numeric_only=True)
            break

    agg = agg.replace([np.inf, -np.inf], np.nan).reset_index().rename(columns={id_col: 'id_jogador'})
    return agg


def explain_risk_reasons(agg: pd.DataFrame) -> dict:
    """Compute simple rule-based explanations per player using robust thresholds."""
    reasons = {}
    if agg.shape[0] == 0:
        return reasons

    numeric_cols = agg.select_dtypes(include=[np.number]).columns.tolist()
    stats = agg[numeric_cols].agg(['median', 'mean', 'std']).to_dict()

    for _, row in agg.iterrows():
        uid = row.get('id_jogador')
        r = []
        # high avg stake
        if 'avg_stake' in row.index and not pd.isna(row['avg_stake']):
            med = stats['avg_stake']['median'] if 'avg_stake' in stats else np.nan
            std = stats['avg_stake']['std'] if 'avg_stake' in stats else 0
            if not pd.isna(med) and row['avg_stake'] > (med + 2 * (std if not pd.isna(std) else 0)):
                r.append('Avg stake muito alto')

        # large total stake
        if 'total_stake' in row.index and not pd.isna(row['total_stake']):
            med = stats['total_stake']['median'] if 'total_stake' in stats else np.nan
            std = stats['total_stake']['std'] if 'total_stake' in stats else 0
            if not pd.isna(med) and row['total_stake'] > (med + 2 * (std if not pd.isna(std) else 0)):
                r.append('Volume total apostado elevado')

        # low return_ratio
        if 'return_ratio' in row.index and not pd.isna(row['return_ratio']):
            if row['return_ratio'] < 0:
                r.append('Perdas sistemáticas (return_ratio < 0)')

        # high odds
        if 'avg_odds' in row.index and not pd.isna(row['avg_odds']):
            if row['avg_odds'] >= 5.0:
                r.append('Odds médias altas')

        # high bet count
        if 'bet_count' in row.index and not pd.isna(row['bet_count']):
            med = stats['bet_count']['median'] if 'bet_count' in stats else np.nan
            std = stats['bet_count']['std'] if 'bet_count' in stats else 0
            if not pd.isna(med) and row['bet_count'] > (med + 2 * (std if not pd.isna(std) else 0)):
                r.append('Alto número de apostas')

        # max deposit
        if 'max_deposit' in row.index and not pd.isna(row['max_deposit']):
            med = stats['max_deposit']['median'] if 'max_deposit' in stats else np.nan
            std = stats['max_deposit']['std'] if 'max_deposit' in stats else 0
            if not pd.isna(med) and row['max_deposit'] > (med + 2 * (std if not pd.isna(std) else 0)):
                r.append('Depósito máximo elevado')

        # fallback reason based on risk score will be computed elsewhere, so keep list
        reasons[uid] = r if len(r) > 0 else ['Padrão incomum detectado']

    return reasons


def train_model():
    # train on training CSV
    if not os.path.exists(TRAINING_CSV):
        raise FileNotFoundError(f'Training CSV not found at {TRAINING_CSV}')
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    agg = aggregate_by_player(df)

    # features (drop id_jogador)
    features = [c for c in agg.columns if c != 'id_jogador']
    X = agg[features].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_cols)], remainder='drop')

    print('Fitting preprocessor...')
    X_prep = preprocessor.fit_transform(X)

    print('Training IsolationForest...')
    iso = IsolationForest(n_estimators=300, contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X_prep)

    artifact = {'preprocessor': preprocessor, 'model': iso, 'features': features, 'numeric_cols': numeric_cols}
    joblib.dump(artifact, MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')
    return artifact


if __name__ == '__main__':
    # when run directly, create app and start
    app = create_app()
    # train if model missing
    if not os.path.exists(MODEL_PATH):
        try:
            train_model()
        except Exception as e:
            print('Warning: training failed', e)
    app.run(host='0.0.0.0', port=int(os.environ.get('SERVER_PORT', 5000)), debug=os.environ.get('FLASK_DEBUG', 'True') == 'True')
import os
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash, jsonify
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

from .models import db, AnalysisResult
from .features import build_feature_table

# Load environment variables
load_dotenv()

# Get base directory
BASEDIR = os.path.dirname(os.path.abspath(__file__))

# Create required directories
REQUIRED_DIRS = [
    'ml_models',
    'instance',
    'uploads',
    'uploads/test_files'
]

for dir_name in REQUIRED_DIRS:
    dir_path = os.path.join(BASEDIR, dir_name)
    os.makedirs(dir_path, exist_ok=True)

# Configuration
UPLOAD_FOLDER = os.path.join(BASEDIR, 'uploads', 'test_files')
INSTANCE_FOLDER = os.path.join(BASEDIR, 'instance')
ML_MODELS_DIR = os.path.join(BASEDIR, 'ml_models')
MODEL_PATH = os.path.join(ML_MODELS_DIR, 'risk_model.pkl')
ALLOWED_EXT = {'.csv'}

# CSV schema validation
REQUIRED_COLUMNS = {'user_id', 'stake_amount', 'odds', 'created_at'}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXT


def validate_csv_schema(df):
    """Validate CSV structure against reference schema."""
    df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        return False, f"Missing required columns: {', '.join(sorted(missing_cols))}"
    return True, None


def create_app(test_config=None):
    """Application factory."""
    app = Flask(__name__, 
                template_folder=os.path.join(BASEDIR, 'templates'),
                static_folder=os.path.join(BASEDIR, 'static'))
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False') == 'True'
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))
    
    # SQLAlchemy configuration
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(INSTANCE_FOLDER, "risk.db")}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database
    db.init_app(app)
    
    with app.app_context():
        db.create_all()
    
    # ==================== ROUTES ====================
    
    @app.route('/')
    def index():
        """Dashboard home page."""
        try:
            data_path = os.path.join(BASEDIR, 'data', 'data_treino', 'dados_juntos.csv')
            if os.path.exists(data_path):
                df = pd.read_csv(data_path, low_memory=False)
                df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
                total_players = df['user_id'].nunique() if 'user_id' in df.columns else 0
                total_bets = df.shape[0]
            else:
                total_players = 0
                total_bets = 0
            
            # Get recent analysis results
            recent = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(5).all()
            
            return render_template('index.html', 
                                 total_players=total_players, 
                                 total_bets=total_bets,
                                 recent_analysis=recent)
        except Exception as e:
            flash(f"Error loading dashboard: {str(e)}", 'error')
            return render_template('index.html', total_players=0, total_bets=0, recent_analysis=[])
    
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        """Handle file upload and batch prediction."""
        if request.method == 'POST':
            # Validate file presence
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
            
            if not file or not allowed_file(file.filename):
                flash('Invalid file format. Only CSV allowed.', 'error')
                return redirect(request.url)
            
            try:
                # Save uploaded file
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                
                # Load and validate CSV
                df = pd.read_csv(save_path, low_memory=False)
                valid, error_msg = validate_csv_schema(df)
                if not valid:
                    flash(f'CSV validation failed: {error_msg}', 'error')
                    os.remove(save_path)
                    return redirect(request.url)
                
                # Normalize columns
                df.columns = [c.strip().lower().replace(' ', '_').replace('-', '_') for c in df.columns]
                
                # Build features
                user_agg = build_feature_table(df)
                results = user_agg[['user_id']].copy()
                
                # Load model and predict
                if os.path.exists(MODEL_PATH):
                    artifact = joblib.load(MODEL_PATH)
                    preprocessor = artifact['preprocessor']
                    model = artifact['model']
                    feat_list = artifact['features']
                    
                    # Select available features
                    feat = [c for c in feat_list if c in user_agg.columns]
                    X = user_agg[feat].fillna(0)
                    X_trans = preprocessor.transform(X)
                    
                    # Score samples (higher = more like training set = risk)
                    raw_scores = model.score_samples(X_trans)
                    
                    # Normalize to 0-1
                    minv = float(raw_scores.min())
                    maxv = float(raw_scores.max())
                    if maxv - minv > 1e-9:
                        norm = (raw_scores - minv) / (maxv - minv)
                    else:
                        norm = np.zeros_like(raw_scores)
                    
                    results['risk_score'] = norm
                else:
                    flash('Model not found. Please train the model first.', 'error')
                    results['risk_score'] = 0.0
                
                # Save results
                out_name = f"results_{os.path.splitext(filename)[0]}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.csv"
                out_path = os.path.join(UPLOAD_FOLDER, out_name)
                results.to_csv(out_path, index=False)
                
                # Compute statistics
                total_users = results.shape[0]
                risk_flagged = (results['risk_score'] >= 0.5).sum()
                avg_risk = results['risk_score'].mean()
                
                # Save to database
                analysis = AnalysisResult(
                    filename=out_name,
                    total_users=total_users,
                    risk_flagged=int(risk_flagged),
                    avg_risk_score=float(avg_risk),
                    file_path=out_path
                )
                db.session.add(analysis)
                db.session.commit()
                
                flash(f'✅ Analysis complete! {risk_flagged}/{total_users} users flagged as risk.', 'success')
                return redirect(url_for('results', filename=out_name))
                
            except Exception as e:
                flash(f'Error processing file: {str(e)}', 'error')
                return redirect(request.url)
        
        return render_template('upload.html')
    
    
    @app.route('/results')
    def results():
        """Display analysis results."""
        filename = request.args.get('filename')
        
        if not filename:
            # Show list of all results
            all_results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).all()
            return render_template('results_list.html', results=all_results)
        
        # Show specific result
        try:
            analysis = AnalysisResult.query.filter_by(filename=filename).first()
            if not analysis:
                flash('Result not found', 'error')
                return redirect(url_for('results'))
            
            # Load results CSV
            df = pd.read_csv(analysis.file_path)
            rows = df.to_dict(orient='records')
            cols = list(df.columns)
            
            # Compute histogram data
            if 'risk_score' in df.columns:
                counts, edges = np.histogram(df['risk_score'].fillna(0).values, bins=20)
                hist_labels = [round(float(e), 3) for e in edges[:-1]]
                hist_counts = counts.tolist()
            else:
                hist_labels = []
                hist_counts = []
            
            return render_template('results.html',
                                 rows=rows,
                                 cols=cols,
                                 filename=filename,
                                 analysis=analysis,
                                 hist_labels=hist_labels,
                                 hist_counts=hist_counts)
        
        except Exception as e:
            flash(f'Error loading results: {str(e)}', 'error')
            return redirect(url_for('results'))
    
    
    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        """Download uploaded results file."""
        return send_from_directory(UPLOAD_FOLDER, filename)
    
    
    @app.route('/api/analysis')
    def api_analysis():
        """API endpoint for analysis statistics."""
        results = AnalysisResult.query.order_by(AnalysisResult.created_at.desc()).limit(10).all()
        return jsonify([r.to_dict() for r in results])
    
    
    return app
