import pytest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, MagicMock
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

def test_tqdm_progress_bar_in_train(mock_ml_data):
    """Verifies that tqdm is called during the training process."""
    engine = ValuationEngine(mode="public")
    X, y = engine.prepare_data(mock_ml_data, target_col="enterprise_value")
    
    with patch("src.valuation.tqdm") as mock_tqdm:
        # Configure mock_tqdm to act like a context manager
        mock_pbar = MagicMock()
        mock_tqdm.return_value.__enter__.return_value = mock_pbar
        
        engine.train(X, y)
        
        # Verify tqdm was initialized
        mock_tqdm.assert_called_once_with(total=1, desc="Training Model")
        # Verify update was called
        mock_pbar.update.assert_called_once_with(1)

def test_dynamic_target_rebuild(mock_ml_data):
    """Verifies that the engine correctly handles changing target columns dynamically."""
    engine = ValuationEngine(mode="public")
    
    # Initial state
    initial_target = engine.target_cols
    
    # Change target to one of the features
    new_target = "forwardPE"
    X, y = engine.prepare_data(mock_ml_data, target_col=new_target)
    
    assert engine.target_cols == [new_target]
    assert new_target not in X.columns
    
    # Verify the pipeline's ColumnTransformer was updated
    preprocessor = engine.pipeline.named_steps['preprocess']
    fin_prep_cols = preprocessor.transformers[0][2]
    assert new_target not in fin_prep_cols
    
    # Should be able to train and predict without "column not found" error
    engine.train(X, y)
    preds = engine.predict(X)
    assert preds.shape == (20, 1)
