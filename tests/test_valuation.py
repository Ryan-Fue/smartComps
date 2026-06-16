import pytest
import pandas as pd
import numpy as np
import os
from src.valuation import ValuationEngine

@pytest.fixture
def mock_ml_data():
    """Generates a mock dataset resembling the Gold layer with flattened embeddings and categorical data."""
    np.random.seed(42)
    n_samples = 100
    
    # Financial data
    data = {
        "ticker": [f"TICK{i}" for i in range(n_samples)],
        "forwardPE": np.random.uniform(10, 30, n_samples),
        "ev_to_ebitda": np.random.uniform(5, 15, n_samples),
        "ebitda": np.random.uniform(100, 1000, n_samples),
        "total_cash": np.random.uniform(50, 200, n_samples),
        "total_debt": np.random.uniform(10, 50, n_samples),
        "employee_count": np.random.uniform(10, 500, n_samples),
        "estimated_revenue": np.random.uniform(1000, 5000, n_samples),
        "enterprise_value": np.random.uniform(500, 2000, n_samples),
        "sector": np.random.choice(["Tech", "Finance", "Healthcare"], n_samples)
    }
    
    # NLP flattened embeddings (768 dims)
    for i in range(768):
        data[f"nlp_{i}"] = np.random.rand(n_samples)
        
    return pd.DataFrame(data)

def test_valuation_engine_dynamic_init():
    """Verifies that the engine initializes with all base columns by default."""
    engine = ValuationEngine()
    assert len(engine.base_fin_cols) == 8
    assert len(engine.base_nlp_cols) == 768
    assert engine.base_cat_cols == ["sector"]

def test_schema_adaptation(mock_ml_data):
    """Verifies that the engine correctly adapts to available features in the dataframe."""
    engine = ValuationEngine()
    
    # Test 1: Full data
    X, y = engine.prepare_data(mock_ml_data)
    # 7 fin + 1 cat + 768 nlp + 14 derived = 790 features
    assert X.shape[1] == 790
    assert "sector" in engine.cat_cols
    
    # Test 2: Missing financial columns
    df_slim = mock_ml_data.drop(columns=["forwardPE", "total_cash"])
    X_slim, y_slim = engine.prepare_data(df_slim)
    # 5 fin + 1 cat + 768 nlp + 14 derived = 788 features
    assert X_slim.shape[1] == 788
    assert "forwardPE" not in engine.fin_cols

def test_train_predict_cycle(mock_ml_data):
    """Verifies the standard train/predict flow with the universal data model."""
    engine = ValuationEngine()
    X, y = engine.prepare_data(mock_ml_data)
    
    engine.train(X, y)
    predictions = engine.predict(X)
    
    assert predictions.shape == (100, 1)
    metrics = engine.evaluate(X, y)
    assert "r2" in metrics
    assert metrics["mae"] >= 0

def test_model_persistence(tmp_path, mock_ml_data):
    """Verifies model saving and loading retains predictive consistency."""
    engine = ValuationEngine()
    X, y = engine.prepare_data(mock_ml_data)
    engine.train(X, y)
    
    model_path = tmp_path / "model.joblib"
    engine.save_model(str(model_path))
    
    assert model_path.exists()
    
    new_engine = ValuationEngine()
    new_engine.load_model(str(model_path))
    
    # We must call prepare_data on new_engine to set up its internal feature lists 
    # OR we can assume it loads the full pipeline correctly. 
    # joblib.load(pipeline) restores the full state.
    
    preds_original = engine.predict(X)
    preds_new = new_engine.predict(X)
    
    np.testing.assert_array_almost_equal(preds_original.values, preds_new.values)

def test_optuna_tuning(mock_ml_data):
    """Verifies that Optuna tuning successfully completes and updates the model."""
    engine = ValuationEngine()
    X, y = engine.prepare_data(mock_ml_data)
    
    initial_n_estimators = engine.n_estimators
    
    # Run a very short optimization (2 trials)
    best_params = engine.tune_hyperparameters(X, y, n_trials=2)
    
    assert isinstance(best_params, dict)
    assert "n_estimators" in best_params
    assert "learning_rate" in best_params
    
    # Verify model is updated
    assert engine.n_estimators == best_params["n_estimators"]
    
    # Verify tuning didn't break prediction
    preds = engine.predict(X)
    assert preds.shape == (100, 1)

def test_infinite_value_handling(mock_ml_data):
    """Verifies that infinite values in the input data are handled correctly."""
    df = mock_ml_data.copy()
    df.loc[0, "forwardPE"] = np.inf
    df.loc[1, "ev_to_ebitda"] = -np.inf
    
    engine = ValuationEngine()
    X, y = engine.prepare_data(df)
    
    # Check that inf values are gone in numeric features
    numeric_X = X.select_dtypes(include=[np.number])
    assert not np.isinf(numeric_X).values.any()
    
    # Replace with NaN check
    assert np.isnan(X.loc[0, "forwardPE"])
    assert np.isnan(X.loc[1, "ev_to_ebitda"])
    
    # Training should still work due to KNNImputer
    engine.train(X, y)
    assert engine.predict(X).shape == (100, 1)
