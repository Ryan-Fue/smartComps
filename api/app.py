from flask import Flask, render_template, request, jsonify
import os
import sys
import pandas as pd
import logging
from sklearn.model_selection import train_test_split

# Set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to sys.path so we can import src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.valuation import ValuationEngine
from src.embedder import FeatureProcessor
from src.data_loader import FinancialDataLoader

app = Flask(__name__)

# Global engine variable to persist the model after training
engine = None

# Global processor for embedding
processor = FeatureProcessor()

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

    data = request.json
    selected = data.get('features', [])
    target = data.get('target', 'enterprise_value')

    # Sort out parameters

    fin_cols = [f for f in selected if f in ValuationEngine.DEFAULT_FIN_COLS]
    cat_cols = [f for f in selected if f in ValuationEngine.DEFAULT_CAT_COLS]
    nlp_cols = list(ValuationEngine.DEFAULT_NLP_COLS) if 'business_summary' in selected else [] # Required but redudant check for future implimentation

    try:
        df = pd.read_parquet('data/processed/UNIVERSAL_embedded.parquet')

        # Intialize engine with user selection
        global engine
        engine = ValuationEngine(
            fin_cols=fin_cols,
            cat_cols=cat_cols,
            nlp_cols=nlp_cols,
            target_cols =[target]
        )

        X, y = engine.prepare_data(df, target_col=target)

        # Split for accurate metrics
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Light tuning and evaluate for metrics
        engine.tune_hyperparameters(X_train, y_train, n_trials=8)
        metrics = engine.evaluate(X_test, y_test)

        # Train final model on full data set
        engine.tune_hyperparameters(X, y, n_trials=20)

        return jsonify({
            "status": "success",
            "metrics": metrics
        })

    except Exception as e:

        logger.error(f"Training failed: {e}")
        
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500 # Internal server error


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
