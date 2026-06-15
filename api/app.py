from flask import Flask, render_template, request, jsonify
import os
import sys

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation import ValuationEngine
from src.data_loader import FinancialDataLoader

app = Flask(__name__)

@app.route('/')
def index():
    """Serves the main dashboard page."""
    return render_template('index.html')

# --- SKELETON ENDPOINTS ---

@app.route('/api/config', methods=['GET'])
def get_config():

    features = ValuationEngine.DEFAULT_FIN_COLS + ValuationEngine.DEFAULT_CAT_COLS
    targets = ValuationEngine.DEFAULT_FIN_COLS + ValuationEngine.DEFAULT_CAT_COLS

    
    return jsonify({"features": features, "targets": targets})

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    TODO: Accept selected features and target, train the model, and return metrics.
    """
    return jsonify({"message": "Endpoint not implemented yet"})

@app.route('/api/predict', methods=['POST'])
def predict_valuation():
    """
    TODO: Accept input data and return a predicted valuation.
    """
    return jsonify({"message": "Endpoint not implemented yet"})

if __name__ == '__main__':
    app.run(debug=True)
