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
        "targets": [col for col in ValuationEngine.DEFAULT_FIN_COLS if col != "employee_count"],
        "nlp_features": ["business_summary"],
        "sector_options": ValuationEngine.VALID_SECTORS
    })

@app.route('/api/train', methods=['POST'])
def train_model():

    try:

        data = request.json
        selected = data.get('features', [])
        target = data.get('target', 'enterprise_value')

        # Sort out parameters

        fin_cols = [f for f in selected if f in ValuationEngine.DEFAULT_FIN_COLS]
        cat_cols = [f for f in selected if f in ValuationEngine.DEFAULT_CAT_COLS]
        nlp_cols = list(ValuationEngine.DEFAULT_NLP_COLS) if 'business_summary' in selected else [] # Required but redudant check for future implimentation
    
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
    global engine, processor

    if engine is None:
        return jsonify({"status": "error", "message": "Model not trained yet"}), 400
    
    try:
        # Convert user input into a single-row DataFrame
        input_data = request.json
        df_input = pd.DataFrame(input_data, index = [0])

        # If the user provided a business summary AND we trained with NLP
        if 'business_summary' in df_input.columns and engine.nlp_cols:
            # Convert into 384 numbers
            df_input = processor.embed_summaries(df_input)
        
        # Run prediction
        prediction_df = engine.predict(df_input)
        val = prediction_df.iloc[0, 0] # Grab the single result -> in this format since might have multiple outputs later

        # Return formatted value
        target_name = engine.target_cols[0]
        is_currency = any(kw in target_name.lower() for kw in ['value', 'total', 'revenue', 'cash', 'debt'])

        if is_currency:
            # Format as: $1,250,000,000
            formatted_val = f"${val:,.0f}"
        else:
            # Format as a regular decimal: 15.42 (e.g., for PE ratios or EBITDA multiples)
            formatted_val = f"{val:,.2f}"
        
        return jsonify({
            "status": "success",
            "valuation": formatted_val
        })

    except Exception as e:

        logger.error(f"Prediction failed: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    app.run(debug=True)
