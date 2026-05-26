import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error


class ValuationEngine:
    """
    The ValuationEngine is responsible for training, evaluating, and deploying 
    machine learning models to predict company valuations based on 
    quantitative financials and qualitative NLP embeddings.
    """

    # Default feature sets for different company types
    PUBLIC_FIN_COLS = ["forwardPE", "ev_to_ebitda", "ebitda", "total_cash", "total_debt"]
    PRIVATE_FIN_COLS = ["employee_count", "estimated_revenue"]

    def __init__(self, 
                 mode: str = "public", 
                 fin_cols: list = None, 
                 nlp_cols: list = None, 
                 target_cols: list = None, 
                 n_estimators: int = 150, 
                 random_state: int = 42) -> None:
        """
        Initializes the ValuationEngine.
        
        Args:
            mode: 'public', 'private', or 'hybrid'. Determines default fin_cols.
            fin_cols: Explicit list of financial columns.
            nlp_cols: Explicit list of NLP embedding columns.
            target_cols: List of target columns to predict.
            n_estimators: Number of trees in the XGBoost model.
            random_state: Seed for reproducibility.
        """
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.random_state = random_state
        self.n_estimators = n_estimators

        # 1. Determine Financial Columns
        if fin_cols is not None:
            self.base_fin_cols = fin_cols
        elif mode == "public":
            self.base_fin_cols = self.PUBLIC_FIN_COLS
        elif mode == "private":
            self.base_fin_cols = self.PRIVATE_FIN_COLS
        else: # hybrid
            self.base_fin_cols = self.PUBLIC_FIN_COLS + self.PRIVATE_FIN_COLS

        # 2. Determine NLP Columns (default to 384 if not specified)
        if nlp_cols is not None:
            self.base_nlp_cols = nlp_cols
        else:
            self.base_nlp_cols = [f"nlp_{i}" for i in range(384)]

        # 3. Determine Target Columns
        self.target_cols = target_cols if target_cols else ["enterprise_value"]
        
        # 4. Initialize features and pipeline
        self._update_features_and_pipeline()

    def _update_features_and_pipeline(self) -> None:
        """Filters base features against target_cols and rebuilds the pipeline."""
        self.fin_cols = [c for c in self.base_fin_cols if c not in self.target_cols]
        self.nlp_cols = [c for c in self.base_nlp_cols if c not in self.target_cols]
        
        # Rebuild the pipeline with updated feature sets
        self.pipeline = self._build_pipeline(self.n_estimators, self.random_state)
        self.model = self.pipeline  # Maintain alias

    def _build_pipeline(self, n_estimators: int, random_state: int = 42) -> Pipeline:
        """Constructs the Scikit-Learn pipeline."""
        
        # Preprocessor for financial and NLP data
        preprocessor = ColumnTransformer([
            ("fin_prep", Pipeline([
                ("imputer", KNNImputer(n_neighbors=5, weights='distance')),
                ("scaler", RobustScaler())
            ]), self.fin_cols),
            ("nlp_prep", PCA(n_components=10, random_state=random_state), self.nlp_cols)
        ])
        
        # Multi-target XGBoost model
        model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.1,
                max_depth=5,
                random_state=random_state,
                n_jobs=-1
            )
        )

        # Combine into master pipeline
        return Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

    def prepare_data(self, df: pd.DataFrame, target_col: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Filters and splits the dataframe into features (X) and target (y).
        If target_col is provided, it updates the internal state if different from current target.
        """
        self.logger.info("Preparing data for modeling...")
        
        requested_targets = [target_col] if target_col else self.target_cols
        
        # Dynamic rebuild if target changes
        if requested_targets != self.target_cols:
            self.logger.info(f"Target changed to {requested_targets}. Rebuilding pipeline...")
            self.target_cols = requested_targets
            self._update_features_and_pipeline()
            
        # Use updated feature lists
        available_fin = [c for c in self.fin_cols if c in df.columns]
        available_nlp = [c for c in self.nlp_cols if c in df.columns]
        
        X = df[available_fin + available_nlp].copy()
        y = df[self.target_cols].copy()

        # 1. Handle infinite values by converting them to NaN for the imputer
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # 2. Cap extremely large values to prevent numerical overflow in downstream models
        # 1e15 (quadrillion) is a safe upper bound for financial metrics
        X = X.clip(lower=-1e15, upper=1e15)
        y = y.clip(lower=-1e15, upper=1e15)
        
        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the valuation model with a final check on feature alignment."""
        self.logger.info(f"Training model on {len(X_train)} samples...")
        
        # Final safety check: Ensure all columns expected by ColumnTransformer are in X_train
        expected_cols = self.fin_cols + self.nlp_cols
        missing_cols = [c for c in expected_cols if c not in X_train.columns]
        if missing_cols:
            raise ValueError(f"X_train is missing columns expected by pipeline: {missing_cols}")

        with tqdm(total=1, desc="Training Model") as pbar:
            self.pipeline.fit(X_train, y_train)
            pbar.update(1)
        self.logger.info("Training complete.")

    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions for new data."""
        self.logger.info(f"Generating predictions for {len(X_new)} samples...")
        preds = self.pipeline.predict(X_new)
        return pd.DataFrame(preds, columns=self.target_cols, index=X_new.index)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
        """Evaluates the model and returns key metrics."""
        preds = self.pipeline.predict(X)
        
        # Handle multi-output metrics
        r2 = r2_score(y, preds)
        mae = mean_absolute_error(y, preds)
        
        metrics = {"r2": r2, "mae": mae}
        self.logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    def save_model(self, file_path: str) -> None:
        """Persists the trained pipeline to disk."""
        self.logger.info(f"Saving model to {file_path}...")
        joblib.dump(self.pipeline, file_path)

    def load_model(self, file_path: str) -> None:
        """Loads a pre-trained pipeline from disk."""
        self.logger.info(f"Loading model from {file_path}...")
        self.pipeline = joblib.load(file_path)
        self.model = self.pipeline # Update alias


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Valuation Engine Module Loaded.")
