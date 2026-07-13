# smartComps: AI-Powered Quantitative Valuation Engine

`smartComps` is an automated, high-performance quantitative pipeline designed to ingest, sanitize, and enrich massive datasets of public company financials and SEC filings. The system synthesizes hard financial multiples with high-dimensional NLP embeddings to predict company valuations with institutional-grade precision.

Full Website Implementation on [BFB Website](https://github.com/Nathangong-coder/BFB-Website-Re-Design)

<img width="445" height="593" alt="image" src="https://github.com/user-attachments/assets/43e070db-ecc8-4828-a625-2002ceaa044e" />
<img width="520" height="340" alt="image" src="https://github.com/user-attachments/assets/d0143bda-272e-40b0-8fbd-1587d62cf78f" />

---

## 🚀 Key Features

- **Schema-Agnostic Modular Pipeline**: Built for flexibility, the engine supports **dynamic configuration** of both inputs and targets.
- **Configurable Target Variables**: The regression objective is not hardcoded. While it defaults to Enterprise Value, the system supports **target variable 
- **High-Dimensional NLP Enrichment**: Utilizes `all-mpnet-base-v2` to vectorize business summaries, capturing qualitative signals often missed by traditional models.
- **Bayesian Hyperparameter Optimization**: Leverages **Optuna** for automated, high-performance tuning of the underlying XGBoost regressor.
- **Fault-Tolerant Data Ingestion**: Robust Bronze-layer harvesting from SEC EDGAR and Yahoo Finance with integrated retry logic and exponential cooldowns.
- **Mathematical Stability**: Implements symmetric log-transformations and `TransformedTargetRegressor` to handle skewed financial distributions and ensure non-negative valuations.

---

## 🏗 Medallion Architecture

### 1. Bronze Layer (Ingestion)
Fetches raw data from SEC EDGAR and Yahoo Finance. Implements fault-tolerant chunking and a Dead Letter Queue (DLQ) for failed tickers.

### 2. Gold Layer (Sanitization)
Consolidates data into strictly typed Parquet files. Applies a **Mid-Market Filter** ($100M - $100B EV) and enforces data quality thresholds (e.g., minimum revenue, summary length).

### 3. Platinum Layer (Valuation Engine)
The core ML pipeline:
*   **Numerical:** Symmetric log-transformation, standard scaling, and KNN Imputation.
*   **NLP:** 768-dimensional vector compression via PCA.
*   **Training:** Bayesian search via Optuna to optimize XGBoost regressors.

---

## 🛠 Installation & Setup

### Prerequisites
*   [Python 3.13+](https://www.python.org/downloads/)
*   [Poetry](https://python-poetry.org/docs/#installation)
*   [Docker](https://www.docker.com/get-started) (optional, for containerization)

### Local Development
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Ryan-Fue/smartComps.git
   cd smartComps
   ```

2. **Install dependencies:**
   ```bash
   poetry install
   ```

3. **Configure Environment:**
   Create a `.env` file in the project root:
   ```env
   SEC_USER_AGENT="Your Name your-email@domain.com"
   ```

4. **Run Tests:**
   ```bash
   poetry run pytest
   ```

---

## 🏃 Running the Pipeline

Before starting the application, you must populate the data layers and train the model.

1. **Ingest & Sanitize (Bronze → Gold):**
   ```bash
   poetry run python src/data_loader.py
   ```

2. **Embed & Feature Engineer (Gold → Platinum):**
   ```bash
   poetry run python src/embedder.py
   ```

3. **Train & Optimize Model:**
   ```bash
   poetry run python src/valuation.py
   ```
   *This will run Bayesian optimization via Optuna and save the final model to `models/valuation_pipeline.joblib`.*

---

## 🖥 Running the Application

### Method 1: Local Development (Poetry)
Once the pipeline has finished and the model is saved:
```bash
poetry run python api/app.py
```
The dashboard will be available at `http://localhost:5000`.


---

## 📊 Directory Structure

*   `api/`: Flask application, templates, and static assets.
*   `src/`: Core logic (Data loading, Embedding, Valuation Engine).
*   `data/`: Data storage (excluded from git, managed by pipeline).
*   `models/`: Serialized model artifacts (`.joblib`).
*   `tests/`: Comprehensive Pytest suite.

---

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

