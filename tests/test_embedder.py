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

@pytest.fixture
def private_sample_df():
    return pd.DataFrame({
        "ticker": ["P1", "P2"],
        "enterprise_value": [1e7, 2e7],
        "employee_count": [50, 150],
        "estimated_revenue": [5e6, 1.5e7],
        "sector": ["Technology", "Healthcare"],
        "business_summary": ["A stealth mode AI startup", "Digital health platform"]
    })



def test_embed_summaries(processor, sample_df):
    # Use a mock or small subset to avoid long download/run time in CI if needed
    # But here we assume local run is fine
    df_emb = processor.embed_summaries(sample_df.head(2))

    # Make list of nlp columns
    nlp_columns = [col for col in df_emb.columns if col.startswith("nlp_")]
    
    assert "nlp_0" in df_emb.columns
    assert "nlp_383" in df_emb.columns
    assert len(nlp_columns) == 384 # Shape for all-MiniLM-L6-v2


def test_drop_extra_columns(processor, sample_df):

    df = processor.drop_extra_columns(sample_df)

    extra_columns = ["ticker", "sector", "industry", "business_summary"]

    for col in extra_columns:
        assert col not in df.columns

def test_process_pipeline_public(processor, sample_df, tmp_path):
    input_path = tmp_path / "PUBLIC_training.parquet"
    output_path = tmp_path / "PUBLIC_embedded.parquet"
    
    sample_df.to_parquet(input_path)
    
    result = processor.process_pipeline(str(input_path), str(output_path))
    
    assert result is True
    assert os.path.exists(output_path)
    df_processed = pd.read_parquet(output_path)
    nlp_cols = ["nlp_" + str(i) for i in range(384)]
    assert set(nlp_cols).issubset(df_processed.columns)
    assert "forwardPE" in df_processed.columns
    assert "ticker" not in df_processed.columns

def test_process_pipeline_private(processor, private_sample_df, tmp_path):
    input_path = tmp_path / "PRIVATE_training.parquet"
    output_path = tmp_path / "PRIVATE_embedded.parquet"
    
    private_sample_df.to_parquet(input_path)
    
    result = processor.process_pipeline(str(input_path), str(output_path))
    
    assert result is True
    assert os.path.exists(output_path)
    df_processed = pd.read_parquet(output_path)
    nlp_cols = ["nlp_" + str(i) for i in range(384)]
    assert set(nlp_cols).issubset(df_processed.columns)
    assert "estimated_revenue" in df_processed.columns
    assert "ticker" not in df_processed.columns
