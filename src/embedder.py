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
        
        # Define project root relative to this file (src/embedder.py)
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.proc_data_dir = os.path.join(self.project_root, "data", "processed")
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
        """
        Drops non-numeric and metadata columns to prepare data for ML modeling.
        """
        if extra_col is None:
            # Default metadata columns to drop
            extra_col = ["ticker", "sector", "industry", "business_summary", "embeddings"]
        
        # Filter for columns that actually exist in the dataframe
        existing_cols_to_drop = [col for col in extra_col if col in df.columns]
        
        # Identify requested columns that are missing (for logging purposes)
        missing_cols = [col for col in extra_col if col not in df.columns]
        for col in missing_cols:
            self.logger.debug(f"Column '{col}' not found; skipping drop.")

        # Drop columns safely
        df = df.drop(columns=existing_cols_to_drop, errors='ignore')
        self.logger.info(f"Successfully dropped {len(existing_cols_to_drop)} columns.")

        return df

    def process_pipeline(self, input_path: str, output_path: str, drop_cols: list = None) -> bool:
        """
        Full pipeline: Read -> Embed -> Drop -> Save.
        """
        try:
            self.logger.info(f"Loading data from {input_path}...")
            df = pd.read_parquet(input_path)
            
            # 1. Embed business summaries
            df_processed = self.embed_summaries(df)

            # 2. Drop Extra Columns
            df_processed = self.drop_extra_columns(df_processed, extra_col=drop_cols)
            
            # 3. Save processed Gold Layer data
            self.logger.info(f"Saving processed data to {output_path}...")
            df_processed.to_parquet(output_path, index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process pipeline: {e}")
            return False
        

if __name__ == "__main__":
    # Setup production logging
    logging.basicConfig(
        level=logging.INFO, 
        filename="log.log", 
        filemode="a", 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    
    # Add console output for real-time monitoring
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    logger = logging.getLogger(__name__)
    processor = FeatureProcessor()
    root = processor.project_root
    
    # 1. PUBLIC PIPELINE: Pure financial valuation
    # Drops sector as it has minimal impact on hard financial multiples
    public_in = os.path.join(root, "data", "processed", "PUBLIC_training.parquet")
    public_out = os.path.join(root, "data", "processed", "PUBLIC_embedded.parquet")
    public_drop = ["ticker", "sector", "business_summary", "employee_count", "estimated_revenue"]
    
    if os.path.exists(public_in):
        success = processor.process_pipeline(public_in, public_out, drop_cols=public_drop)
        if success:
            logger.info(f"Public pipeline complete -> {public_out}")
    else:
        logger.warning(f"Public input {public_in} not found. Run data_loader.py first.")

    # 2. PRIVATE PIPELINE: Proxy valuation
    # Drops sector as private input data is often noisy or missing these headers
    private_in = os.path.join(root, "data", "processed", "PRIVATE_training.parquet")
    private_out = os.path.join(root, "data", "processed", "PRIVATE_embedded.parquet")
    private_drop = ["ticker", "sector", "business_summary"]
    
    if os.path.exists(private_in):
        success = processor.process_pipeline(private_in, private_out, drop_cols=private_drop)
        if success:
            logger.info(f"Private pipeline complete -> {private_out}")
    else:
        logger.warning(f"Private input {private_in} not found. Run data_loader.py first.")
