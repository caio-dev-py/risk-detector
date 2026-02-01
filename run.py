import os
from dotenv import load_dotenv
from src.risk_detector_ai.app import create_app

# Load environment variables
load_dotenv()

app = create_app()

if __name__ == '__main__':
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True') == 'True'
    
    app.run(host=host, port=port, debug=debug)