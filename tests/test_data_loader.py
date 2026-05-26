import pytest
import pandas as pd
import os
import json
from unittest.mock import MagicMock, patch
from src.data_loader import FinancialDataLoader

@pytest.fixture
def data_loader():
    return FinancialDataLoader()

def test_init(data_loader):
    # Tests assume running from project root
    assert os.path.exists(data_loader.raw_data_dir)
    assert os.path.exists(data_loader.proc_data_dir)

@patch("src.data_loader.requests.get")
def test_download_file_success(mock_get, data_loader, tmp_path):
    # Setup
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value = mock_response
    
    # Override directories for testing
    data_loader.raw_data_dir = str(tmp_path)
    filename = "test.json"
    
    # Execute
    result = data_loader.download_file("http://example.com", filename)
    
    # Verify
    assert result is True
    assert os.path.exists(os.path.join(tmp_path, filename))
    with open(os.path.join(tmp_path, filename), "rb") as f:
        assert f.read() == b"chunk1chunk2"

@patch("src.data_loader.yf.Ticker")
def test_fetch_single_comp_metrics(mock_ticker, data_loader):
    # Setup
    mock_info = {
        "forwardPE": 20.0,
        "enterpriseToEbitda": 15.0,
        "sector": "Technology",
        "industry": "Software",
        "longBusinessSummary": "Test summary " * 10, # Longer than 50 chars
        "ebitda": 1000000,
        "totalCash": 500000,
        "totalDebt": 200000,
        "enterpriseValue": 5000000,
        "fullTimeEmployees": 100,
        "totalRevenue": 2000000
    }
    mock_ticker.return_value.info = mock_info
    
    # Execute
    metrics = data_loader._fetch_single_comp_metrics("AAPL")
    
    # Verify
    assert metrics["ticker"] == "AAPL"
    assert metrics["forwardPE"] == 20.0
    assert metrics["enterprise_value"] == 5000000
    assert len(metrics["business_summary"]) > 50

def test_build_public_training_table(data_loader, tmp_path):
    # Setup dummy master CSV
    master_csv = tmp_path / "master.csv"
    output_parquet = tmp_path / "public.parquet"
    
    df = pd.DataFrame({
        "ticker": ["AAPL", "MISSING", "SHORT"],
        "enterprise_value": [100, None, 100],
        "ebitda": [10, 10, 10],
        "total_debt": [5, 5, 5],
        "total_cash": [2, 2, 2],
        "forwardPE": [15, 15, 15],
        "ev_to_ebitda": [8, 8, 8],
        "sector": ["Tech", "Tech", "Tech"],
        "business_summary": ["A long business summary that should pass the length check" * 2, "Long enough summary", "Short summary"]
    })
    df.to_csv(master_csv, index=False)
    
    # Execute
    result = data_loader.build_public_training_table(str(master_csv), str(output_parquet))
    
    # Verify
    assert result is True
    assert os.path.exists(output_parquet)
    df_public = pd.read_parquet(output_parquet)
    
    # Ticker AAPL stays.
    # MISSING is dropped (enterprise_value is None).
    # SHORT is dropped (summary < 50).
    assert len(df_public) == 1
    assert df_public.iloc[0]["ticker"] == "AAPL"
    assert "forwardPE" in df_public.columns
    assert "employee_count" not in df_public.columns

def test_build_private_training_table(data_loader, tmp_path):
    # Setup dummy master CSV
    master_csv = tmp_path / "master.csv"
    output_parquet = tmp_path / "private.parquet"
    
    df = pd.DataFrame({
        "ticker": ["PVT1", "MISSING"],
        "enterprise_value": [100, None],
        "employee_count": [50, 50],
        "estimated_revenue": [1000, 1000],
        "sector": ["Tech", "Tech"],
        "business_summary": ["A long business summary for the private training table test" * 2, "Long enough"]
    })
    df.to_csv(master_csv, index=False)
    
    # Execute
    result = data_loader.build_private_training_table(str(master_csv), str(output_parquet))
    
    # Verify
    assert result is True
    assert os.path.exists(output_parquet)
    df_private = pd.read_parquet(output_parquet)
    
    assert len(df_private) == 1
    assert df_private.iloc[0]["ticker"] == "PVT1"
    assert "estimated_revenue" in df_private.columns
    assert "forwardPE" not in df_private.columns
