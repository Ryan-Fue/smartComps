import pandas as pd
import numpy as np
import yfinance as yf
import requests 
import os
import logging
import time
from tqdm import tqdm

class FinancialDataLoader:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Paths are now relative to project root
        self.raw_data_dir = "data/raw"
        self.proc_data_dir = "data/processed"

        # Make sure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.proc_data_dir, exist_ok=True)
        

    # Downloads files from the internet
    # REQUIRES a user header to get SEC filings
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
                "enterprise_value": info.get("enterpriseValue"), # The universal target (y)
                
                # PUBLIC ENGINE FEATURES
                "forwardPE": info.get("forwardPE"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ebitda": ebitda,
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                
                # PRIVATE ENGINE FEATURES (Proxies)
                "employee_count": info.get("fullTimeEmployees"),
                "estimated_revenue": info.get("totalRevenue"),
                
                # UNIVERSAL NLP FEATURES 
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
        
        # Check if a partial file already exists to resume
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

            
            # Save the chunk to the CSV 
            if success:
                chunk_df = pd.DataFrame(chunk_data)

                # Logic to account for missing chunk data
                successful_tickers = chunk_df['ticker'].tolist()
                missed_tickers = [t for t in chunk if t not in successful_tickers]
                
                # Dead letter queue (DQL)
                if missed_tickers:
                    dlq_df = pd.DataFrame({"failed_tickers": missed_tickers})
                    dlq_csv = "data/processed/missed_tickers.csv"
                    dlq_df.to_csv(dlq_csv, mode='a', header=not os.path.exists(dlq_csv), index=False)
                    self.logger.info(f"Chunk {i//chunk_size}: {len(missed_tickers)} tickers missing. Saved to DLQ.")

                # If file exists, append without headers. Otherwise, write new
                chunk_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                time.sleep(10) # Safety pause

            else:
                self.logger.error(f"CRITICAL: Chunk {i//chunk_size} failed after {max_attempts} attempts. Skipping chunk to keep pipeline alive.")
                dlq_df = pd.DataFrame({"failed_tickers": chunk})
                dlq_csv = "data/processed/missed_tickers.csv"
                dlq_df.to_csv(dlq_csv, mode='a', header=not os.path.exists(dlq_csv), index=False)

        return True


    def build_public_training_table(self, master_csv: str, output_parquet: str) -> bool:
        """Filters the master data strictly for the Public Comparables Engine"""

        self.logger.info("Building Public Engine Training Table...")
        df = pd.read_csv(master_csv)

        # Select only columns relevant to public valuation
        public_cols = [
            "ticker", "enterprise_value", "forwardPE", "ev_to_ebitda", 
            "ebitda", "total_cash", "total_debt", "sector", "business_summary"
        ]
        df_public = df[public_cols].copy()

        # Strict dropping for public metrics
        # If a public company doesn't report EBITDA or Debt, we drop it from training.
        df_public = df_public.dropna(subset=["enterprise_value", "ebitda", "total_debt", "business_summary"])
        df_public = df_public[df_public['business_summary'].str.len() > 50]
        
        # Enforce numeric types
        for col in ["enterprise_value", "forwardPE", "ev_to_ebitda", "ebitda", "total_cash", "total_debt"]:
            df_public[col] = pd.to_numeric(df_public[col], errors='coerce')

        df_public = df_public.reset_index(drop=True)
        df_public.to_parquet(output_parquet, index=False)
        self.logger.info(f"Public Training Table complete: {len(df_public)} rows.")
        return True


    def build_private_training_table(self, master_csv: str, output_parquet: str) -> bool:
        """Filters the master data strictly for the Private Startup Engine."""
        self.logger.info("Building Private Engine Training Table...")
        df = pd.read_csv(master_csv)

        # Select only the proxy columns and the universal target
        private_cols = [
            "ticker", "enterprise_value", "employee_count", "estimated_revenue", 
            "sector", "business_summary"
        ]
        df_private = df[private_cols].copy()

        # Strict dropping for private metrics
        # If we don't have the proxy data, the model can't learn the relationship.
        df_private = df_private.dropna(subset=["enterprise_value", "employee_count", "estimated_revenue", "business_summary"])
        df_private = df_private[df_private['business_summary'].str.len() > 50]

        # Enforce numeric types
        for col in ["enterprise_value", "employee_count", "estimated_revenue"]:
            df_private[col] = pd.to_numeric(df_private[col], errors='coerce')

        df_private = df_private.reset_index(drop=True)
        df_private.to_parquet(output_parquet, index=False)
        self.logger.info(f"Private Training Table complete: {len(df_private)} rows.")
        return True


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, 
        filename = "log.log", 
        filemode = "w", 
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

    loader = FinancialDataLoader()

    # SEC Download logic
    sec_url = "https://www.sec.gov/files/company_tickers.json"
    headers = {"User-Agent": "Ryan Fue rfue29@gmail.com"}
    loader.download_file(sec_url, "company_tickers.json", headers=headers)

    master_file = "data/raw/master_metrics.csv"
    raw_tickers_path = "data/raw/company_tickers.json"

    # Use a small limit for verification
    loader.build_raw_master_table(raw_tickers_path, master_file, limit=5)

    loader.build_public_training_table(master_file, "data/processed/PUBLIC_training.parquet")
    loader.build_private_training_table(master_file, "data/processed/PRIVATE_training.parquet")
