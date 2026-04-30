import pytest
import pandas as pd
import numpy as np
import os
from src.embedder import FeatureProcessor

@pytest.fixture
def processor():
    return FeatureProcessor()

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "ticker": ["A", "B", "C"],
        "forwardPE": [10.0, 20.0, 100.0], # 100 is an outlier
        "ev_to_ebitda": [5.0, 10.0, 15.0],
        "ebitda_margin": [0.1, 0.2, 0.3],
        "ebitda": [1e6, 2e6, 3e6],
        "total_cash": [1e5, 2e5, 3e5],
        "total_debt": [5e5, 6e5, 7e5],
        "shares_outstanding": [1e6, 1e6, 1e6],
        "debt_to_ebitda": [0.5, 0.3, 0.23],
        "business_summary": ["Technology company focused on AI", "Retailer with many stores", "Healthcare provider"]
    })

def test_normalize_metrics(processor, sample_df):
    df_norm = processor.normalize_metrics(sample_df)
    
    assert "forwardPE" in df_norm.columns
    # Check that values have changed
    assert not np.array_equal(df_norm["forwardPE"].values, sample_df["forwardPE"].values)
    # RobustScaler scales based on IQR, so it handles outliers well

def test_embed_summaries(processor, sample_df):
    # Use a mock or small subset to avoid long download/run time in CI if needed
    # But here we assume local run is fine
    df_emb = processor.embed_summaries(sample_df.head(2))
    
    assert "embeddings" in df_emb.columns
    assert len(df_emb.iloc[0]["embeddings"]) == 384 # Shape for all-MiniLM-L6-v2

def test_process_pipeline(processor, sample_df, tmp_path):
    input_path = tmp_path / "input.parquet"
    output_path = tmp_path / "output.parquet"
    
    sample_df.to_parquet(input_path)
    
    result = processor.process_pipeline(str(input_path), str(output_path))
    
    assert result is True
    assert os.path.exists(output_path)
    df_processed = pd.read_parquet(output_path)
    assert "embeddings" in df_processed.columns
    assert "forwardPE" in df_processed.columns
