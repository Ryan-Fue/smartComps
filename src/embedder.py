import pandas as pd
import numpy as np
import os
import logging
from sklearn.preprocessing import RobustScaler
from sentence_transformers import SentenceTransformer

class FeatureProcessor:
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the FeatureProcessor with a scaling strategy and a transformer model.
        """
        self.logger = logging.getLogger(__name__)
        self.scaler = RobustScaler()
        self.model = SentenceTransformer(model_name)
        
        self.proc_data_dir = "data/processed"
        os.makedirs(self.proc_data_dir, exist_ok=True)

    def normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies RobustScaler to numerical financial metrics to handle outliers.
        Handles infinities and NaNs by replacing them with column medians.
        """
        self.logger.info("Normalizing numerical metrics using RobustScaler...")
        
        numeric_cols = [
            "forwardPE", "ev_to_ebitda", "ebitda_margin", 
            "ebitda", "total_cash", "total_debt", 
            "shares_outstanding", "debt_to_ebitda"
        ]
        
        # Ensure all columns exist
        existing_cols = [col for col in numeric_cols if col in df.columns]
        
        if not existing_cols:
            self.logger.warning("No numeric columns found for normalization.")
            return df
            
        df_scaled = df.copy()
        
        # Replace inf with NaN then fill with median
        for col in existing_cols:
            df_scaled[col] = df_scaled[col].replace([np.inf, -np.inf], np.nan)
            median_val = df_scaled[col].median()
            df_scaled[col] = df_scaled[col].fillna(median_val)
            
        df_scaled[existing_cols] = self.scaler.fit_transform(df_scaled[existing_cols])
        
        self.logger.info(f"Successfully normalized {len(existing_cols)} columns.")
        return df_scaled

    def embed_summaries(self, df: pd.DataFrame, text_col: str = "business_summary") -> pd.DataFrame:
        """
        Generates text embeddings for the business summary column.
        """
        self.logger.info(f"Generating embeddings for column: {text_col}...")
        
        if text_col not in df.columns:
            self.logger.error(f"Column '{text_col}' not found in DataFrame.")
            return df
            
        summaries = df[text_col].fillna("").tolist()
        embeddings = self.model.encode(summaries, show_progress_bar=True)
        
        # Convert embeddings to a list of arrays to store in DataFrame
        df["embeddings"] = list(embeddings)
        
        self.logger.info("Successfully generated embeddings.")
        return df

    def process_pipeline(self, input_path: str, output_path: str) -> bool:
        """
        Full pipeline: Read -> Normalize -> Embed -> Save.
        """
        try:
            self.logger.info(f"Loading data from {input_path}...")
            df = pd.read_parquet(input_path)
            
            # 1. Normalize
            df_processed = self.normalize_metrics(df)
            
            # 2. Embed
            df_processed = self.embed_summaries(df_processed)
            
            # 3. Save
            self.logger.info(f"Saving processed data to {output_path}...")
            df_processed.to_parquet(output_path, index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process pipeline: {e}")
            return False

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, 
        filename="log.log", 
        filemode="a", 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Adding console output for debugging
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    processor = FeatureProcessor()
    
    input_file = "data/processed/company_metrics_clean.parquet"
    output_file = "data/processed/company_metrics_ml.parquet"
    
    if os.path.exists(input_file):
        processor.process_pipeline(input_file, output_file)
    else:
        print(f"Input file {input_file} not found. Please run data_loader.py first.")
