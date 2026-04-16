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
        
        self.raw_data_dir = "../data/raw"
        self.proc_data_dir = "../data/processed"

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

            return {
                "ticker": ticker,
                
                # Quantitative block for ML
                "forwardPE": info.get("forwardPE"),
                "ev_to_ebitda": info.get("enterpriseToEbitda"),
                "ebitda_margin": info.get("ebitdaMargins"),
                "debt_to_equity": info.get("debtToEquity"),
                
                # Qualitative block for ML
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "business_summary": info.get("longBusinessSummary", ""),
                
                # Data for valuation
                "ebitda": info.get("ebitda"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "shares_outstanding": info.get("sharesOutstanding")
            }
        except Exception as e:
                self.logger.warning(f"Failed to fetch data for {ticker}: {e}")
                return None

    # Builds raw company metrics table
    def build_csv_comps_table(self,raw_data_path: str, output_csv: str, chunk_size: int = 50, limit: int = None) -> bool:

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
                    dlq_csv = "../data/processed/missed_tickers.csv"
                    dlq_df.to_csv(dlq_csv, mode='a', header=not os.path.exists(dlq_csv), index=False)
                    self.logger.info(f"Chunk {i//chunk_size}: {len(missed_tickers)} tickers missing. Saved to DLQ.")

                # If file exists, append without headers. Otherwise, write new
                chunk_df.to_csv(output_csv, mode='a', header=not os.path.exists(output_csv), index=False)
                time.sleep(10) # Safety pause

            else:
                self.logger.error(f"CRITICAL: Chunk {i//chunk_size} failed after {max_attempts} attempts. Skipping chunk to keep pipeline alive.")
                dlq_df = pd.DataFrame({"failed_tickers": chunk})
                dlq_csv = "../data/processed/missed_tickers.csv"
                dlq_df.to_csv(dlq_csv, mode='a', header=not os.path.exists(dlq_csv), index=False)

        return True


    def clean_csv_comps_table(self, raw_data_path: str, output_parquet: str) -> bool:

        df = pd.read_csv(raw_data_path)

        # Irrecoverable data
        df = df.dropna(subset=["ebitda", "sector", "industry", "shares_outstanding", "business_summary"])
        df["total_debt"] = df["total_debt"].fillna(0)
        df["total_cash"] = df["total_cash"].fillna(0)

        # Adding more robust debt_to_ebitda metric
        df["debt_to_ebitda"] = df["total_debt"] / df["ebitda"]
        df = df.drop(columns=["debt_to_equity"])

        # Conditional fill to account for ev_to_ebita ratios that are because of negative EBDITA vs just missing
        df.loc[df["ebitda"] <= 0, "ev_to_ebitda"] = df.loc[df["ebitda"] <= 0, "ev_to_ebitda"].fillna(0)
        df["ev_to_ebitda"] = df.groupby("sector")["ev_to_ebitda"].transform(lambda x: x.fillna(x.median()))
        df["forwardPE"] = df.groupby("sector")["forwardPE"].transform(lambda x: x.fillna(x.median()))
        df = df.dropna(subset=["ev_to_ebitda", "forwardPE"])

        df = df.reset_index(drop=True)

        df.to_parquet(output_parquet, index=False)

        return True



if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO, 
        filename = "log.log", 
        filemode = "w", 
        format = "%(asctime)s - %(levelname)s - %(message)s"
    )

    loader = FinancialDataLoader()

    loader.build_csv_comps_table("../data/raw/company_tickers.json", "../data/raw/company_metrics.csv", 100)