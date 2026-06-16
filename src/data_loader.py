import pandas as pd
import numpy as np
import yfinance as yf
import requests 
import os
from dotenv import load_dotenv
import logging
import time
from tqdm import tqdm

load_dotenv()

class FinancialDataLoader:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define project root relative to this file
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Paths are now absolute to project root
        self.raw_data_dir = os.path.join(self.project_root, "data", "raw")
        self.proc_data_dir = os.path.join(self.project_root, "data", "processed")

        # Make sure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.proc_data_dir, exist_ok=True)
        

    # Downloads files from the internet
    def download_file(self, url: str, filename: str, headers: dict = {}) -> bool:
        save_path = os.path.join(self.raw_data_dir, filename)

        if os.path.exists(save_path):
            self.logger.info(f"{filename} already exists. Skipping download.")
            return True
        
        try:
            self.logger.info(f"Downloading {filename} from web...")
            response = requests.get(url, headers = headers, stream = True)
            response.raise_for_status()

            with open(save_path, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            self.logger.info(f"Successfully saved to {save_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            return False
    

   # Helper function to build raw company metric table
    def _fetch_single_comp_metrics(self,ticker: str) -> dict:

        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            # Safely calculate EBITDA if missing
            ebitda = info.get("ebitda")
            if ebitda is None:
                ebitda = info.get("operatingIncome", 0)

            return {
                "ticker": ticker,
                "enterprise_value": info.get("enterpriseValue"), 
                
                # ALL FEATURES
                "forwardPE": info.get("forwardPE"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ebitda": ebitda,
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "employee_count": info.get("fullTimeEmployees"),
                "estimated_revenue": info.get("totalRevenue"),
                
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "business_summary": info.get("longBusinessSummary", "")
            }
        except Exception as e:
                self.logger.warning(f"Failed to fetch data for {ticker}: {e}")
                return None

    # Builds raw company metrics table
    def build_raw_master_table(self,raw_data_path: str, output_csv: str, chunk_size: int = 50, limit: int = None) -> bool:

        self.logger.info("Starting large data pull...")
        df_raw = pd.read_json(raw_data_path, orient='index')
        all_tickers = df_raw['ticker'].tolist()

        if limit:
            all_tickers = all_tickers[:limit]
            self.logger.info(f"Test Mode: Only processing the first {limit} companies")
        
        start_index = 0
        if os.path.exists(output_csv):
            existing_df = pd.read_csv(output_csv)
            start_index = len(existing_df)
            self.logger.info(f"Found existing file. Resuming from ticker {start_index}...")

        for i in range(start_index, len(all_tickers), chunk_size):
            chunk = all_tickers[i : i + chunk_size]
            max_attempts = 3
            attempt = 0
            success = False

            while attempt < max_attempts and not success:
                chunk_data = []
                for ticker in tqdm(chunk, desc=f"Chunk {i//chunk_size}", leave=False):
                    metrics = self._fetch_single_comp_metrics(ticker)
                    if metrics:
                        chunk_data.append(metrics)

                if len(chunk_data) == 0:
                    attempt += 1
                    self.logger.warning(f"Chunk {i//chunk_size} failed. Attempt {attempt}/{max_attempts}. Entering 60s cooldown...")
                    time.sleep(60)
                else:
                    success = True

            if success:
                chunk_df = pd.DataFrame(chunk_data)
                successful_tickers = chunk_df['ticker'].tolist()
                missed_tickers = [t for t in chunk if t not in successful_tickers]
                
                if missed_tickers:
                    dlq_df = pd.DataFrame({"failed_tickers": missed_tickers})
                    dlq_csv = os.path.join(self.proc_data_dir, "missed_tickers.csv")
                    dlq_df.to_csv(dlq_csv, mode='a', header=not os.path.exists(dlq_csv), index=False)

                chunk_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                time.sleep(5) # Safety pause
            else:
                self.logger.error(f"CRITICAL: Chunk failed. Skipping.")
                dlq_df = pd.DataFrame({"failed_tickers": chunk})
                dlq_csv = os.path.join(self.proc_data_dir, "missed_tickers.csv")
                dlq_df.to_csv(dlq_csv, mode='a', header=not os.path.exists(dlq_csv), index=False)

        return True


    def build_universal_training_table(self, master_csv: str, output_parquet: str) -> bool:
        """Sanitizes raw master table for bad data and prepares for universal modeling."""

        self.logger.info("Building Universal Training Table...")
        df = pd.read_csv(master_csv)

        # Enforce numeric types for all quantitative columns
        num_cols = ["enterprise_value", "forwardPE", "ev_to_ebitda", "ebitda", "total_cash", "total_debt", "estimated_revenue", "employee_count"]
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Strictly drop rows missing critical anchors
        df = df.dropna(subset=["business_summary", "enterprise_value", "estimated_revenue"])
        df = df[df['business_summary'].str.len() > 50]

        # Drop negative revenue and tiny revenue (Speculative/Bad data)
        df = df[df['estimated_revenue'] >= 10_000_000]

        # MID-MARKET FILTER: Target Universe: $100M to $100B Enterprise Value
        min_ev = 100_000_000
        max_ev = 100_000_000_000
        initial_count = len(df)
        df = df[(df['enterprise_value'] >= min_ev) & (df['enterprise_value'] <= max_ev)]
        
        # Consistency Check: Remove companies with extreme Revenue-to-EV ratios
        # These are usually distressed firms or data errors that confuse the model
        df['rev_to_ev'] = df['estimated_revenue'] / (df['enterprise_value'] + 1)
        df = df[df['rev_to_ev'] < 10] # Drop if revenue is 10x larger than EV
        df = df.drop(columns=['rev_to_ev'])

        removed = initial_count - len(df)
        self.logger.info(f"Filtering complete: Removed {removed} noisy/outlier rows.")

        df = df.reset_index(drop=True)
        df.to_parquet(output_parquet, index=False)
        self.logger.info(f"Universal Training Table complete: {len(df)} rows.")
        return True


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, 
        filename = "log.log", 
        filemode = "w", 
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

    loader = FinancialDataLoader()
    root = loader.project_root

    # SEC Download logic
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    sec_user_agent = os.getenv("SEC_USER_AGENT")
    headers = {"User-Agent": sec_user_agent}
    loader.download_file(sec_url, "company_tickers.json", headers=headers)

    master_file = os.path.join(root, "data", "raw", "master_metrics.csv")
    raw_tickers_path = os.path.join(root, "data", "raw", "company_tickers.json")

    # Full data pull (set limit=None for production)
    # loader.build_raw_master_table(raw_tickers_path, master_file, limit=None)

    # Rebuild the universal table from the existing master CSV
    loader.build_universal_training_table(master_file, os.path.join(root, "data", "processed", "UNIVERSAL_training.parquet"))
