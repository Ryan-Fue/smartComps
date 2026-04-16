import pandas as pd
import numpy as np
import yfinance as yf
import requests 
import os
import logging

class FinancialDataLoader:
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        self.raw_data_dir = "../data/raw"
        self.proc_data_dir = "../data/processed"

        # Make sure directories exist
        os.makedirs(self.raw_data_dir, exist_ok=True)
        os.makedirs(self.proc_data_dir, exist_ok=True)
        

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
        
    if __name__ == "data_loader.py":

        loader = FinancialDataLoader()