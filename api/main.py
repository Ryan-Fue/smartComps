from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from src.valuation import ValuationEngine
from src.embedder import FeatureProcessor
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="smartComps API",
    description="AI-Powered Quantitative Valuation Engine API",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Core Engines
# NOTE: In a production environment, you would load a pre-trained model from disk.
engine = ValuationEngine(mode="public")
processor = FeatureProcessor()

# Optional: Path to a saved model
MODEL_PATH = os.path.join(os.getcwd(), "models", "public_valuation_model.joblib")

@app.on_event("startup")
async def load_model():
    if os.path.exists(MODEL_PATH):
        logger.info(f"Loading pre-trained model from {MODEL_PATH}...")
        engine.load_model(MODEL_PATH)
    else:
        logger.warning("No pre-trained model found at startup. Predictions will use an untrained pipeline.")

class ValuatorInput(BaseModel):
    forwardPE: float = Field(..., example=25.5, description="Forward Price-to-Earnings Ratio")
    ev_to_ebitda: float = Field(..., example=12.2, description="Enterprise Value to EBITDA Ratio")
    ebitda: float = Field(..., example=500000000, description="Earnings Before Interest, Taxes, Depreciation, and Amortization")
    total_cash: float = Field(..., example=100000000, description="Total Cash and Short-term Investments")
    total_debt: float = Field(..., example=250000000, description="Total Debt")
    business_summary: str = Field(..., example="A global leader in sustainable energy solutions...", description="Long-form business description for NLP embedding")

class ValuationOutput(BaseModel):
    estimated_enterprise_value: float
    currency: str = "USD"

@app.post("/predict", response_model=ValuationOutput)
async def predict_valuation(input_data: ValuatorInput):
    """
    Takes financial metrics and a business summary, processes them through 
    the NLP embedder and XGBoost regressor, and returns an estimated valuation.
    """
    try:
        # 1. Convert Pydantic model to DataFrame for processing
        raw_df = pd.DataFrame([input_data.dict()])
        
        # 2. Enrich with NLP Embeddings (384-dimensional vector)
        logger.info("Generating NLP embeddings for business summary...")
        enriched_df = processor.embed_summaries(raw_df)
        
        # 3. Generate Prediction
        # The engine handles internal scaling, imputer, and PCA dimensionality reduction
        prediction_df = engine.predict(enriched_df)
        
        # 4. Extract scalar value
        estimated_val = float(prediction_df.values[0][0])
        
        return ValuationOutput(estimated_enterprise_value=estimated_val)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing valuation request.")

@app.get("/health")
async def health_check():
    """Returns the status of the API and model readiness."""
    return {
        "status": "online",
        "model_loaded": os.path.exists(MODEL_PATH),
        "engine_mode": engine.mode
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
