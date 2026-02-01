import os
from flask import Flask
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def create_app(test_config=None):
    pkg_dir = os.path.dirname(__file__)
    app = Flask(__name__, template_folder=os.path.join(pkg_dir, 'templates'), static_folder=os.path.join(pkg_dir, 'static'))

    # Load configuration from environment variables
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['ENV'] = os.getenv('FLASK_ENV', 'production')
    app.config['DEBUG'] = os.getenv('FLASK_DEBUG', 'False') == 'True'
    app.config['MAX_CONTENT_LENGTH'] = int(os.getenv('MAX_CONTENT_LENGTH', 16777216))

    # register routes blueprint
    from . import routes as routes_mod
    app.register_blueprint(routes_mod.bp)

    return app
