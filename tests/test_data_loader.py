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

@patch("src.data_loader.yf.Ticker")
def test_fetch_single_comp_metrics_ebitda_fallback(mock_ticker, data_loader):
    # Setup: ebitda is missing, operatingIncome is present
    mock_info = {
        "operatingIncome": 800000,
        "enterpriseValue": 5000000,
        "sector": "Technology",
        "longBusinessSummary": "Test summary " * 10
    }
    mock_ticker.return_value.info = mock_info
    
    # Execute
    metrics = data_loader._fetch_single_comp_metrics("AAPL")
    
    # Verify
    assert metrics["ebitda"] == 800000
    assert metrics["enterprise_value"] == 5000000

def test_build_universal_training_table(data_loader, tmp_path):
    # Setup dummy master CSV
    master_csv = tmp_path / "master.csv"
    output_parquet = tmp_path / "universal.parquet"
    
    df = pd.DataFrame({
        "ticker": ["KEEP1", "MISSING_EV", "MISSING_EBITDA", "MISSING_REV", "SHORT_SUMMARY", "KEEP2"],
        "enterprise_value": [500_000_000, None, 500_000_000, 500_000_000, 500_000_000, 1_000_000_000],
        "ebitda": [50_000_000, 50_000_000, None, 50_000_000, 50_000_000, 100_000_000],
        "estimated_revenue": [100_000_000, 100_000_000, 1_000_000, None, 100_000_000, 200_000_000],
        "total_debt": [5, 5, 5, 5, 5, 10],
        "total_cash": [2, 2, 2, 2, 2, 4],
        "forwardPE": [15, 15, 15, 15, 15, 30],
        "ev_to_ebitda": [8, 8, 8, 8, 8, 16],
        "employee_count": [100, 100, 100, 100, 100, 200],
        "sector": ["Tech", "Tech", "Tech", "Tech", "Tech", "Finance"],
        "business_summary": [
            "A very long business summary that should pass the length check and is definitely more than fifty characters." * 2, 
            "Valid length summary for missing EV test case...", 
            "Valid length summary for missing EBITDA test case...",
            "Valid length summary for missing Revenue test case...",
            "Short summary",
            "Another valid and long summary for the second keeper company in this test suite."
        ]
    })
    df.to_csv(master_csv, index=False)
    
    # Execute
    result = data_loader.build_universal_training_table(str(master_csv), str(output_parquet))
    
    # Verify
    assert result is True
    assert os.path.exists(output_parquet)
    df_universal = pd.read_parquet(output_parquet)
    
    # KEEP1 and KEEP2 stay.
    # MISSING_EV is dropped.
    # MISSING_EBITDA is dropped.
    # MISSING_REV is dropped.
    # SHORT_SUMMARY is dropped.
    assert len(df_universal) == 2
    assert list(df_universal["ticker"]) == ["KEEP1", "KEEP2"]
    
    # Check numeric types
    assert pd.api.types.is_numeric_dtype(df_universal["enterprise_value"])
    assert pd.api.types.is_numeric_dtype(df_universal["ebitda"])
    assert pd.api.types.is_numeric_dtype(df_universal["estimated_revenue"])
