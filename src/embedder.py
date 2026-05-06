import pandas as pd
import numpy as np
import os
import logging
from sentence_transformers import SentenceTransformer

class FeatureProcessor:
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initializes the FeatureProcessor with a scaling strategy and a transformer model.
        """
        self.logger = logging.getLogger(__name__)
        self.model = SentenceTransformer(model_name)
        
        self.proc_data_dir = "data/processed"
        os.makedirs(self.proc_data_dir, exist_ok=True)

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
        
        # Make flattened embedding dataframe
        embedding_df = pd.DataFrame(list(embeddings), index=df.index)
        embedding_df = embedding_df.add_prefix("nlp_")
        self.logger.info(f"Generated columns for: {text_col}...")

        # Merge embedding_df with main DataFrame
        df = df.join(embedding_df)
        
        self.logger.info("Successfully appended embeddings.")
        return df
    

    def drop_extra_columns(self, df: pd.DataFrame, extra_col: list = None) -> pd.DataFrame:
        if extra_col is None:
            extra_col = ["ticker", "sector", "industry", "business_summary", "embeddings"]
        
        missing_cols = [col for col in extra_col if col not in df.columns]
        for col in missing_cols:
            self.logger.warning(f"Failed to find {col}")

        # 2. Safely drop whatever is in the list without crashing
        df = df.drop(columns=extra_col, errors='ignore')

        self.logger.info("Successfully dropped extra columns.")

        return df

    def process_pipeline(self, input_path: str, output_path: str) -> bool:
        """
        Full pipeline: Read -> Embed -> Save.
        """
        try:
            self.logger.info(f"Loading data from {input_path}...")
            df = pd.read_parquet(input_path)
            
            # 1. Embed
            df_processed = self.embed_summaries(df)

            # 2. Drop Extra Columns
            df_processed = self.drop_extra_columns(df_processed)
            
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
    
    # Ensure paths are relative to the project root
    input_file = "data/processed/company_metrics_clean.parquet"
    output_file = "data/processed/company_metrics_ml.parquet"
    
    if os.path.exists(input_file):
        processor.process_pipeline(input_file, output_file)
        print(f"Pipeline complete. Processed data saved to {output_file}")
    else:
        print(f"Input file {input_file} not found. Please run data_loader.py first.")
