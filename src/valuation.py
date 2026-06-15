import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import optuna
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, TargetEncoder, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error
from sklearn.model_selection import KFold, cross_val_score


"""
These helper Functions are used by the class to preform safe scaling.
"""

def safe_expm1(x):
    """Inverse log transform that guarantees results >= 0 for mathematical stability."""
    return np.clip(np.expm1(x), 0, None)

def sym_log_transform(x):
    """Symmetric log transform to handle negative values while compressing scale."""
    return np.sign(x) * np.log1p(np.abs(x))

class ValuationEngine:
    """
    The ValuationEngine is responsible for training, evaluating, and deploying 
    machine learning models to predict company valuations based on 
    quantitative financials and qualitative NLP embeddings.
    """

    # --- Class Constants (Read-only defaults) ---
    DEFAULT_FIN_COLS = (
        "forwardPE", "ev_to_ebitda", "ebitda", "total_cash", 
        "total_debt", "employee_count", "estimated_revenue", "enterprise_value"
    )
    DEFAULT_CAT_COLS = ("sector",)
    DEFAULT_TARGET_COLS = ("enterprise_value",)
    DEFAULT_NLP_COLS = tuple(f"nlp_{i}" for i in range(384))

    VALID_SECTORS = (
        "Basic Materials",
        "Communication Services",
        "Consumer Cyclical",
        "Consumer Defensive",
        "Energy",
        "Financial Services",
        "Healthcare",
        "Industrials",
        "Real Estate",
        "Technology",
        "Unknown",
        "Utilities"
    )

    def __init__(self, 
                 fin_cols: list = None, 
                 nlp_cols: list = None, 
                 cat_cols: list = None,
                 target_cols: list = None, 
                 n_estimators: int = 100, 
                 random_state: int = 42) -> None:
        """Initializes the ValuationEngine with dynamic feature and target support."""
        self.logger = logging.getLogger(__name__)
        self.random_state = random_state
        self.n_estimators = n_estimators

        # Default column sets if not provided (Convert tuples to lists for internal mutation if needed)
        self.base_fin_cols = list(fin_cols) if fin_cols is not None else list(self.DEFAULT_FIN_COLS)
        self.base_nlp_cols = list(nlp_cols) if nlp_cols is not None else list(self.DEFAULT_NLP_COLS)
        self.base_cat_cols = list(cat_cols) if cat_cols is not None else list(self.DEFAULT_CAT_COLS)
        self.target_cols = list(target_cols) if target_cols is not None else list(self.DEFAULT_TARGET_COLS)
        
        # Initialize current features (will be refined in prepare_data)
        self.fin_cols = []
        self.nlp_cols = []
        self.cat_cols = []
        
        # Build initial pipeline
        self._update_pipeline()


    def _update_pipeline(self) -> None:
        """Rebuilds the Scikit-Learn pipeline based on current feature lists."""
        transformers = []
        
        if self.fin_cols:
            transformers.append(("fin_prep", Pipeline([
                ("sym_log", FunctionTransformer(sym_log_transform)),
                ("scaler", StandardScaler()),
                ("imputer", KNNImputer(n_neighbors=5, weights='distance'))
            ]), self.fin_cols))
            
        if self.cat_cols:
            transformers.append(("cat_prep", Pipeline([
                ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
                ("encoder", TargetEncoder(target_type='continuous', random_state=self.random_state))
            ]), self.cat_cols))
            
        if self.nlp_cols:
            transformers.append(("nlp_prep", Pipeline([
                ("imputer", SimpleImputer(strategy='mean')),
                ("pca", PCA(n_components=min(50, len(self.nlp_cols)), random_state=self.random_state))
            ]), self.nlp_cols))

        if not transformers:
            # Fallback if no features provided
            preprocessor = "passthrough"
        else:
            preprocessor = ColumnTransformer(transformers)
        
        # Base XGBoost model
        base_model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=0.05,
            max_depth=5,
            random_state=self.random_state,
            n_jobs=-1
        )

        # Wrap in TransformedTargetRegressor to handle log scaling for the target
        self.pipeline = Pipeline([
            ('preprocess', preprocessor),
            ('model', TransformedTargetRegressor(
                regressor=base_model,
                func=np.log1p,
                inverse_func=safe_expm1
            ))
        ])

    def prepare_data(self, df: pd.DataFrame, target_col: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Dynamically adapts the engine to the provided dataframe's available features and target."""
        self.logger.info("Preparing data for modeling...")
        
        # 1. Create Derived Features (Ratios)
        if "ebitda" in self.base_fin_cols and "estimated_revenue" in self.base_fin_cols:
            df["ebitda_margin"] = df["ebitda"] / (df["estimated_revenue"] + 1)
            if "ebitda_margin" not in self.base_fin_cols:
                self.base_fin_cols.append("ebitda_margin")

        if "total_debt" in self.base_fin_cols and "estimated_revenue" in self.base_fin_cols:
            df["debt_to_revenue"] = df["total_debt"] / (df["estimated_revenue"] + 1)
            if "debt_to_revenue" not in self.base_fin_cols:
                self.base_fin_cols.append("debt_to_revenue")

        requested_targets = [target_col] if target_col else self.target_cols
        available_targets = [c for c in requested_targets if c in df.columns]
        
        if not available_targets:
             raise ValueError(f"None of the requested targets {requested_targets} found in dataframe.")

        # Update target if it changed
        if requested_targets != self.target_cols:
            self.target_cols = requested_targets

        # Dynamically identify available features (excluding the target)
        available_fin = [c for c in self.base_fin_cols if c in df.columns and c not in self.target_cols]
        available_nlp = [c for c in self.base_nlp_cols if c in df.columns and c not in self.target_cols]
        available_cat = [c for c in self.base_cat_cols if c in df.columns and c not in self.target_cols]
        
        # Rebuild pipeline if feature schema changed
        if (set(available_fin) != set(self.fin_cols) or 
            set(available_nlp) != set(self.nlp_cols) or 
            set(available_cat) != set(self.cat_cols)):
            self.logger.info("Schema change detected. Rebuilding pipeline...")
            self.fin_cols = available_fin
            self.nlp_cols = available_nlp
            self.cat_cols = available_cat
            self._update_pipeline()

        X = df[self.fin_cols + self.cat_cols + self.nlp_cols].copy()
        y = df[self.target_cols].copy()

        # Handle extreme values and NaNs in target
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        nan_targets = y.isna().any(axis=1)
        if nan_targets.any():
            self.logger.warning(f"Dropping {nan_targets.sum()} rows with NaN targets.")
            X = X[~nan_targets]
            y = y[~nan_targets]
        
        # Numerical features clip to prevent overflow/unstable math
        num_cols = self.fin_cols + self.nlp_cols
        if num_cols:
            X[num_cols] = X[num_cols].clip(lower=-1e13, upper=1e13)
        
        y = y.clip(lower=0, upper=1e13) 
        
        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the valuation model."""
        self.logger.info(f"Training model on {len(X_train)} samples with {X_train.shape[1]} features...")
        self.pipeline.fit(X_train, y_train)

    def tune_hyperparameters(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.DataFrame, 
                             n_trials: int = 50, 
                             timeout: int = 600) -> dict:
        """Uses Optuna to find the best hyperparameters and automatically updates the pipeline."""
        self.logger.info(f"Starting Optuna optimization (n_trials={n_trials})...")

        def objective(trial):
            # Define hyperparameter search space
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10)
            }
            
            # Temporary pipeline for validation
            # We bypass TransformedTargetRegressor in inner cross-val for speed if needed, 
            # but for accuracy we keep the same structure.
            regressor = xgb.XGBRegressor(**params, random_state=self.random_state, n_jobs=-1)
            model = TransformedTargetRegressor(
                regressor=regressor,
                func=np.log1p,
                inverse_func=safe_expm1
            )
            
            val_pipeline = Pipeline([
                ('preprocess', self.pipeline.named_steps['preprocess']),
                ('model', model)
            ])
            
            # Cross-validation
            score = cross_val_score(
                val_pipeline, X_train, y_train, 
                cv=KFold(n_splits=3, shuffle=True, random_state=self.random_state),
                scoring="neg_root_mean_squared_log_error",
                n_jobs=-1
            ).mean()
            
            return score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        self.logger.info(f"Optimization complete. Best score: {study.best_value}")
        
        # Update engine with best params and retrain
        self.n_estimators = study.best_params['n_estimators']
        self._update_pipeline()
        
        # Set the rest of the xgb params directly into the regressor
        best_xgb_params = {k: v for k, v in study.best_params.items() if k != 'n_estimators'}
        self.pipeline.named_steps['model'].regressor.set_params(**best_xgb_params)
        
        # Final retrain on full data
        self.logger.info("Retraining model with best hyperparameters...")
        self.train(X_train, y_train)
        
        self.logger.info("Pipeline updated and fully trained with best parameters.")
        return study.best_params

    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions and clips them to realistic valuation bounds."""
        X_pred = X_new.copy()
        
        if "ebitda" in X_pred.columns and "estimated_revenue" in X_pred.columns:
            X_pred["ebitda_margin"] = X_pred["ebitda"] / (X_pred["estimated_revenue"] + 1)

        if "total_debt" in X_pred.columns and "estimated_revenue" in X_pred.columns:
            X_pred["debt_to_revenue"] = X_pred["total_debt"] / (X_pred["estimated_revenue"] + 1)

        preds = self.pipeline.predict(X_pred)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        
        preds = np.clip(preds, 0, 1e13)
        return pd.DataFrame(preds, columns=self.target_cols, index=X_new.index)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
        """Evaluates the model and returns robust relative and absolute metrics."""
        preds_df = self.predict(X)
        preds = preds_df.values
        y_val = y.values
        
        abs_pct_error = np.abs((y_val - preds) / (y_val + 1))
        y_val_clipped = np.clip(y_val, 0, None)
        
        metrics = {
            "r2": r2_score(y_val, preds),
            "mae": mean_absolute_error(y_val, preds),
            "mape": mean_absolute_percentage_error(y_val + 1, preds + 1),
            "mdape": np.median(abs_pct_error),
            "accuracy_20": np.mean(abs_pct_error < 0.20),
            "rmsle": root_mean_squared_log_error(y_val_clipped, preds)
        }
        self.logger.info(f"Evaluation Metrics: {metrics}")
        return metrics

    def save_model(self, file_path: str) -> None:
        """Persists model to disk."""
        joblib.dump(self.pipeline, file_path)

    def load_model(self, file_path: str) -> None:
        """Loads model from disk."""
        self.pipeline = joblib.load(file_path)

if __name__ == "__main__":
    import os
    from sklearn.model_selection import train_test_split
    
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
    
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(project_root, "data", "processed", "UNIVERSAL_embedded.parquet")
    model_dir = os.path.join(project_root, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "valuation_pipeline.joblib")

    if os.path.exists(input_path):
        logger.info(f"Loading embedded data from {input_path}...")
        df = pd.read_parquet(input_path)
        
        # Initialize Engine with NLP and Core Features
        engine = ValuationEngine(
            fin_cols=["forwardPE", "ebitda", "total_cash", "total_debt", "estimated_revenue"],
            cat_cols=["sector"],
            nlp_cols=list(ValuationEngine.DEFAULT_NLP_COLS)
        )
        
        # Prepare Data (This will automatically generate ebitda_margin and debt_to_revenue)
        X, y = engine.prepare_data(df)
        
        # Split data for evaluation
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Hyperparameter Tuning
        logger.info("Starting hyperparameter tuning (30 trials)...")
        engine.tune_hyperparameters(X_train, y_train, n_trials=30)
        
        # Final Evaluation
        logger.info("Evaluating optimized model...")
        metrics = engine.evaluate(X_test, y_test)
        logger.info(f"Final Test Metrics: {metrics}")
        
        # Save Model
        logger.info(f"Saving model to {model_path}...")
        engine.save_model(model_path)
        logger.info("Universal Valuation Model applied and saved successfully.")
        
    else:
        logger.warning(f"Embedded input {input_path} not found. Ensure src/data_loader.py and src/embedder.py have been run.")


