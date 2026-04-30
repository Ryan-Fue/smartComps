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
    assert os.path.exists(data_loader.raw_data_dir)
    assert os.path.exists(data_loader.proc_data_dir)

@patch("src.data_loader.requests.get")
def test_download_file_success(mock_get, data_loader, tmp_path):
    # Setup
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.iter_content.return_value = [b"chunk1", b"chunk2"]
    mock_get.return_value = mock_response
    
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
        "ebitdaMargins": 0.25,
        "debtToEquity": 50.0,
        "sector": "Technology",
        "industry": "Software",
        "longBusinessSummary": "Test summary",
        "ebitda": 1000000,
        "totalCash": 500000,
        "totalDebt": 200000,
        "sharesOutstanding": 100000
    }
    mock_ticker.return_value.info = mock_info
    
    # Execute
    metrics = data_loader._fetch_single_comp_metrics("AAPL")
    
    # Verify
    assert metrics["ticker"] == "AAPL"
    assert metrics["forwardPE"] == 20.0
    assert metrics["sector"] == "Technology"
    assert metrics["business_summary"] == "Test summary"

def test_clean_csv_comps_table(data_loader, tmp_path):
    # Setup dummy CSV
    csv_path = tmp_path / "raw.csv"
    parquet_path = tmp_path / "clean.parquet"
    
    df = pd.DataFrame({
        "ticker": ["AAPL", "MSFT"],
        "ebitda": [100, 200],
        "sector": ["Tech", "Tech"],
        "industry": ["Soft", "Soft"],
        "shares_outstanding": [10, 20],
        "business_summary": ["A", "B"],
        "total_debt": [50, 100],
        "total_cash": [20, 40],
        "forwardPE": [15, 25],
        "ev_to_ebitda": [10, 12],
        "debt_to_equity": [0.5, 0.6]
    })
    df.to_csv(csv_path, index=False)
    
    # Execute
    result = data_loader.clean_csv_comps_table(str(csv_path), str(parquet_path))
    
    # Verify
    assert result is True
    assert os.path.exists(parquet_path)
    df_clean = pd.read_parquet(parquet_path)
    assert "debt_to_ebitda" in df_clean.columns
    assert "debt_to_equity" not in df_clean.columns
