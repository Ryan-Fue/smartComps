import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from tqdm import tqdm
from scipy.stats import loguniform, randint
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler, TargetEncoder, FunctionTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_log_error
from sklearn.model_selection import RandomizedSearchCV, KFold


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

    # Default feature sets for different company types
    PUBLIC_FIN_COLS = ["forwardPE", "ev_to_ebitda", "ebitda", "total_cash", "total_debt"]
    PRIVATE_FIN_COLS = ["employee_count", "estimated_revenue"]
    CAT_COLS = ["sector"]

    def __init__(self, 
                 mode: str = "public", 
                 fin_cols: list = None, 
                 nlp_cols: list = None, 
                 cat_cols: list = None,
                 target_cols: list = None, 
                 n_estimators: int = 150, 
                 random_state: int = 42) -> None:
        """Initializes the ValuationEngine with multi-mode support."""
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        self.random_state = random_state
        self.n_estimators = n_estimators

        # 1. Determine Base Financial Columns
        if fin_cols is not None:
            self.base_fin_cols = fin_cols
        elif mode == "public":
            self.base_fin_cols = self.PUBLIC_FIN_COLS
        elif mode == "private":
            self.base_fin_cols = self.PRIVATE_FIN_COLS
        else: # hybrid
            self.base_fin_cols = self.PUBLIC_FIN_COLS + self.PRIVATE_FIN_COLS

        # 2. Determine Base NLP Columns
        if nlp_cols is not None:
            self.base_nlp_cols = nlp_cols
        else:
            self.base_nlp_cols = [f"nlp_{i}" for i in range(384)]

        # 3. Determine Base Categorical Columns
        self.base_cat_cols = cat_cols if cat_cols is not None else self.CAT_COLS

        # 4. Determine Target Columns
        self.target_cols = target_cols if target_cols else ["enterprise_value"]
        
        # 5. Initialize features and pipeline
        self._update_features_and_pipeline()

    def _update_features_and_pipeline(self) -> None:
        """Filters base features against target_cols and rebuilds the pipeline."""
        self.fin_cols = [c for c in self.base_fin_cols if c not in self.target_cols]
        self.nlp_cols = [c for c in self.base_nlp_cols if c not in self.target_cols]
        self.cat_cols = [c for c in self.base_cat_cols if c not in self.target_cols]
        
        # Rebuild the pipeline
        self.pipeline = self._build_pipeline(self.n_estimators, self.random_state)
        self.model = self.pipeline

    def _build_pipeline(self, n_estimators: int, random_state: int = 42) -> Pipeline:
        """Constructs the Scikit-Learn pipeline with symmetric log-transformed features."""
        
        # Preprocessor for financial, categorical, and NLP data
        preprocessor = ColumnTransformer([
            ("fin_prep", Pipeline([
                ("sym_log", FunctionTransformer(sym_log_transform)),
                ("scaler", StandardScaler()),
                ("imputer", KNNImputer(n_neighbors=5, weights='distance'))
            ]), self.fin_cols),
            ("cat_prep", Pipeline([
                ("imputer", SimpleImputer(strategy='constant', fill_value='Unknown')),
                ("encoder", TargetEncoder(target_type='continuous', random_state=random_state))
            ]), self.cat_cols),
            ("nlp_prep", Pipeline([
                ("imputer", SimpleImputer(strategy='mean')),
                ("pca", PCA(n_components=30, random_state=random_state))
            ]), self.nlp_cols)
        ])
        
        # Base XGBoost model
        # We slightly increase depth back to 5 but keep min_child_weight for stability
        base_model = MultiOutputRegressor(
            xgb.XGBRegressor(
                n_estimators=n_estimators,
                learning_rate=0.05,
                max_depth=5,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=random_state,
                n_jobs=-1
            )
        )

        # Wrap in TransformedTargetRegressor to handle log scaling (log1p/safe_expm1)
        model = TransformedTargetRegressor(
            regressor=base_model,
            func=np.log1p,
            inverse_func=safe_expm1
        )

        return Pipeline([
            ('preprocess', preprocessor),
            ('model', model)
        ])

    def prepare_data(self, df: pd.DataFrame, target_col: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filters and cleans data for modeling, enforcing positive targets for log compatibility."""
        self.logger.info("Preparing data for modeling...")
        
        requested_targets = [target_col] if target_col else self.target_cols
        
        if requested_targets != self.target_cols:
            self.logger.info(f"Target changed to {requested_targets}. Rebuilding pipeline...")
            self.target_cols = requested_targets
            self._update_features_and_pipeline()
            
        available_fin = [c for c in self.fin_cols if c in df.columns]
        available_nlp = [c for c in self.nlp_cols if c in df.columns]
        available_cat = [c for c in self.cat_cols if c in df.columns]
        
        X = df[available_fin + available_cat + available_nlp].copy()
        y = df[self.target_cols].copy()

        # Handle infinite and extreme values
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        y.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Drop rows where target is NaN
        nan_targets = y.isna().any(axis=1)
        if nan_targets.any():
            self.logger.warning(f"Dropping {nan_targets.sum()} rows with NaN targets.")
            X = X[~nan_targets]
            y = y[~nan_targets]
        
        # Clip numerical features to realistic financial bounds
        num_cols = available_fin + available_nlp
        X[num_cols] = X[num_cols].clip(lower=-1e13, upper=1e13)
        
        # Targets must be positive for log1p. We also cap at 10 Trillion to prevent outliers.
        y = y.clip(lower=0, upper=1e13) 
        
        return X, y

    def train(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> None:
        """Trains the valuation model."""
        self.logger.info(f"Training model on {len(X_train)} samples...")
        
        expected_cols = self.fin_cols + self.cat_cols + self.nlp_cols
        available_cols = [c for c in expected_cols if c in X_train.columns]
        X_train = X_train[available_cols]

        with tqdm(total=1, desc="Training Model") as pbar:
            self.pipeline.fit(X_train, y_train)
            pbar.update(1)

    def tune_hyperparameters(self, 
                             X_train: pd.DataFrame, 
                             y_train: pd.DataFrame, 
                             param_distributions: dict = None, 
                             n_iter: int = 50, 
                             cv: int = 5, 
                             scoring: str = "neg_mean_squared_log_error",
                             n_jobs: int = -1) -> dict:
        """Conducts hyperparameter tuning using randomized search."""
        self.logger.info(f"Tuning hyperparameters (n_iter={n_iter}, cv={cv}, n_jobs={n_jobs})...")

        if param_distributions is None:
            param_distributions = {
                "preprocess__nlp_prep__pca__n_components": randint(10, 100),
                "preprocess__fin_prep__imputer__n_neighbors": randint(3, 15),
                "model__regressor__estimator__max_depth": randint(3, 10),
                "model__regressor__estimator__learning_rate": loguniform(1e-3, 0.3),
                "model__regressor__estimator__n_estimators": randint(100, 1000),
                "model__regressor__estimator__subsample": loguniform(0.6, 1.0),
                "model__regressor__estimator__colsample_bytree": loguniform(0.6, 1.0)
            }

        search = RandomizedSearchCV(
            estimator=self.pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=KFold(n_splits=cv, shuffle=True, random_state=self.random_state),
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=n_jobs,
            verbose=1
        )

        with tqdm(total=1, desc="Tuning Hyperparameters") as pbar:
            search.fit(X_train, y_train)
            pbar.update(1)

        self.pipeline = search.best_estimator_
        self.model = self.pipeline
        return search.best_params_

    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        """Generates predictions and clips them to realistic valuation bounds."""
        preds = self.pipeline.predict(X_new)
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)
        
        # Clip final predictions to max realistic global valuation (10 Trillion)
        preds = np.clip(preds, 0, 1e13)
        return pd.DataFrame(preds, columns=self.target_cols, index=X_new.index)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame) -> dict[str, float]:
        """Evaluates the model and returns robust relative and absolute metrics."""
        preds_df = self.predict(X)
        preds = preds_df.values
        y_val = y.values
        
        # Calculate relative errors
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
        self.model = self.pipeline

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Valuation Engine Module Upgraded.")
