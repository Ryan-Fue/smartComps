import pandas as pd
import numpy as np
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.abspath("src"))
from valuation import ValuationEngine

logging.basicConfig(level=logging.INFO)

data_path = "data/processed/PUBLIC_embedded.parquet"
if os.path.exists(data_path):
    df = pd.read_parquet(data_path)
    df = df[df['enterprise_value'] > 0].head(100) # Small sample for speed
    
    engine = ValuationEngine(mode="public", n_estimators=10)
    X, y = engine.prepare_data(df, target_col="enterprise_value")
    
    print("X columns:", X.columns.tolist()[:10])
    print("X cat cols:", engine.cat_cols)
    print("y type:", type(y))
    print("y shape:", y.shape)

    try:
        engine.train(X, y)
        print("Train successful!")
    except Exception as e:
        print("\n!!! TRAIN FAILED !!!")
        import traceback
        traceback.print_exc()
else:
    print(f"Data not found at {data_path}")
