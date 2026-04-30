import pytest
import pandas as pd
import numpy as np
from src.valuation import ValuationEngine

@pytest.fixture
def mock_ml_data():
    """Generates a mock dataset resembling the 'Gold' layer."""
    np.random.seed(42)
    n_samples = 20
    data = {
        "ticker": [f"TICK{i}" for i in range(n_samples)],
        "forwardPE": np.random.uniform(10, 30, n_samples),
        "ev_to_ebitda": np.random.uniform(5, 15, n_samples),
        "ebitda": np.random.uniform(100, 1000, n_samples),
        "embeddings": [np.random.rand(384) for _ in range(n_samples)]
    }
    return pd.DataFrame(data)

def test_valuation_engine_initialization():
    engine = ValuationEngine()
    assert engine.model is not None
    assert engine.random_state == 42

def test_data_preparation(mock_ml_data):
    engine = ValuationEngine()
    X, y = engine.prepare_data(mock_ml_data, target_col="forwardPE")
    
    # 2 numeric features + 384 embeddings = 386 features
    assert X.shape == (20, 386)
    assert y.shape == (20,)

def test_train_predict_cycle(mock_ml_data):
    engine = ValuationEngine()
    X, y = engine.prepare_data(mock_ml_data)
    
    engine.train(X, y)
    predictions = engine.predict(X)
    
    assert predictions.shape == (20,)
    # R2 should be decent on training data for a RF
    metrics = engine.evaluate(X, y)
    assert metrics["r2"] > 0.5 

def test_model_persistence(tmp_path, mock_ml_data):
    engine = ValuationEngine()
    X, y = engine.prepare_data(mock_ml_data)
    engine.train(X, y)
    
    model_path = tmp_path / "model.joblib"
    engine.save_model(str(model_path))
    
    assert model_path.exists()
    
    new_engine = ValuationEngine()
    new_engine.load_model(str(model_path))
    
    preds_original = engine.predict(X)
    preds_new = new_engine.predict(X)
    
    np.testing.assert_array_almost_equal(preds_original, preds_new)
