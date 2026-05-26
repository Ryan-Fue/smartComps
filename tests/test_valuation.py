import pytest
import pandas as pd
import numpy as np
import os
from src.valuation import ValuationEngine

@pytest.fixture
def mock_ml_data():
    """Generates a mock dataset resembling the Gold layer with flattened embeddings."""
    np.random.seed(42)
    n_samples = 20
    
    # Financial data
    data = {
        "ticker": [f"TICK{i}" for i in range(n_samples)],
        "forwardPE": np.random.uniform(10, 30, n_samples),
        "ev_to_ebitda": np.random.uniform(5, 15, n_samples),
        "ebitda": np.random.uniform(100, 1000, n_samples),
        "total_cash": np.random.uniform(50, 200, n_samples),
        "total_debt": np.random.uniform(10, 50, n_samples),
        "enterprise_value": np.random.uniform(500, 2000, n_samples)
    }
    
    # NLP flattened embeddings (384 dims)
    for i in range(384):
        data[f"nlp_{i}"] = np.random.rand(n_samples)
        
    return pd.DataFrame(data)

def test_valuation_engine_initialization():
    engine = ValuationEngine()
    assert engine.model is not None
    assert engine.random_state == 42
    assert engine.mode == "public"

def test_data_preparation(mock_ml_data):
    engine = ValuationEngine(mode="public")
    X, y = engine.prepare_data(mock_ml_data, target_col="forwardPE")
    
    # Public features: ["forwardPE", "ev_to_ebitda", "ebitda", "total_cash", "total_debt"]
    # If forwardPE is target, remaining fin features are 4.
    # Plus 384 NLP features = 388 features.
    assert X.shape == (20, 388)
    assert y.shape == (20, 1)

def test_train_predict_cycle(mock_ml_data):
    engine = ValuationEngine(mode="public")
    X, y = engine.prepare_data(mock_ml_data, target_col="enterprise_value")
    
    engine.train(X, y)
    predictions = engine.predict(X)
    
    assert predictions.shape == (20, 1)
    metrics = engine.evaluate(X, y)
    assert metrics["r2"] > 0.5 

def test_model_persistence(tmp_path, mock_ml_data):
    engine = ValuationEngine(mode="public")
    X, y = engine.prepare_data(mock_ml_data, target_col="enterprise_value")
    engine.train(X, y)
    
    model_path = tmp_path / "model.joblib"
    engine.save_model(str(model_path))
    
    assert model_path.exists()
    
    new_engine = ValuationEngine(mode="public")
    new_engine.load_model(str(model_path))
    
    preds_original = engine.predict(X)
    preds_new = new_engine.predict(X)
    
    np.testing.assert_array_almost_equal(preds_original.values, preds_new.values)

def test_private_mode_initialization():
    engine = ValuationEngine(mode="private")
    assert "employee_count" in engine.fin_cols
    assert "estimated_revenue" in engine.fin_cols
    assert "forwardPE" not in engine.fin_cols
