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
        "forwardPE": [10.0, 20.0, 100.0],
        "ev_to_ebitda": [5.0, 10.0, 15.0],
        "ebitda": [1e6, 2e6, 3e6],
        "total_cash": [1e5, 2e5, 3e5],
        "total_debt": [5e5, 6e5, 7e5],
        "estimated_revenue": [1e7, 2e7, 3e7],
        "employee_count": [100, 200, 300],
        "sector": ["Technology", "Healthcare", "Finance"],
        "industry": ["Software", "Pharma", "Banking"],
        "business_summary": ["Technology company focused on AI", "Retailer with many stores", "Healthcare provider"]
    })

def test_embed_summaries(processor, sample_df):
    # Test embedding generation
    df_emb = processor.embed_summaries(sample_df.head(2))

    # Check for NLP columns
    nlp_columns = [col for col in df_emb.columns if col.startswith("nlp_")]
    
    assert "nlp_0" in df_emb.columns
    assert "nlp_383" in df_emb.columns
    assert len(nlp_columns) == 384 

def test_drop_extra_columns(processor, sample_df):
    # Test that metadata is dropped but features like 'sector' are kept
    df = processor.drop_extra_columns(sample_df)

    # These should be gone
    assert "ticker" not in df.columns
    assert "industry" not in df.columns
    assert "business_summary" not in df.columns
    
    # This must stay for ValuationEngine
    assert "sector" in df.columns
    
    # Numerical features stay
    assert "forwardPE" in df.columns

def test_process_pipeline_universal(processor, sample_df, tmp_path):
    # Test full pipeline logic
    input_path = tmp_path / "UNIVERSAL_training.parquet"
    output_path = tmp_path / "UNIVERSAL_embedded.parquet"
    
    sample_df.to_parquet(input_path)
    
    result = processor.process_pipeline(str(input_path), str(output_path))
    
    assert result is True
    assert os.path.exists(output_path)
    df_processed = pd.read_parquet(output_path)
    
    # Verify embeddings exist
    nlp_cols = ["nlp_" + str(i) for i in range(384)]
    assert set(nlp_cols).issubset(df_processed.columns)
    
    # Verify column management
    assert "sector" in df_processed.columns
    assert "ticker" not in df_processed.columns
    assert "industry" not in df_processed.columns
