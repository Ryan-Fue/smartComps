from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation import ValuationEngine
from src.data_loader import FinancialDataLoader

app = Flask(__name__)

# Global engine variable to persist the model after training
engine = None

@app.route('/')
def index():
    """Serves the main dashboard page."""
    return render_template('index.html')

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        "financial_features": ValuationEngine.DEFAULT_FIN_COLS,
        "categorical_features": ValuationEngine.DEFAULT_CAT_COLS,
        "targets": ValuationEngine.DEFAULT_FIN_COLS + ValuationEngine.DEFAULT_CAT_COLS,
        "nlp_features": ["business_summary"],
        "sector_options": ValuationEngine.VALID_SECTORS
    })

@app.route('/api/train', methods=['POST'])
def train_model():
    """
    TODO: 
    1. Capture 'features' and 'target' from request.json.
    2. Map those features to fin_cols, cat_cols, and nlp_cols.
    3. Load 'data/processed/UNIVERSAL_training.parquet' using pandas.
    4. Instantiate the global 'engine' and run engine.train(df).
    5. Return the resulting metrics as JSON.
    """
    return jsonify({"message": "Endpoint not implemented yet"})

@app.route('/api/predict', methods=['POST'])
def predict_valuation():
    """
    TODO: 
    1. Check if the global 'engine' exists.
    2. Convert request.json into a single-row pandas DataFrame.
    3. (Advanced) If business_summary is used, embed it using FeatureProcessor.
    4. Run engine.predict(df_input) and return the result.
    """
    return jsonify({"message": "Endpoint not implemented yet"})

if __name__ == '__main__':
    app.run(debug=True)
