"""
Fraud detection ML tool using XGBoost.
Trains on synthetic data if no pre-trained model exists.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "fraud_model.joblib"
)

# Feature names expected by the model
FEATURE_NAMES = [
    "amount_claimed",
    "num_procedures",
    "claim_frequency_provider",
    "days_between_claims",
    "is_weekend_claim",
    "amount_vs_avg_ratio",
    "num_diagnosis_codes",
    "procedure_complexity_score",
]


def _generate_synthetic_training_data(n_samples: int = 2000) -> pd.DataFrame:
    """
    Generate synthetic training data for the fraud model.

    Args:
        n_samples: Number of rows to generate.

    Returns:
        DataFrame with feature columns and 'is_fraud' label.
    """
    rng = np.random.default_rng(42)
    n_fraud = int(n_samples * 0.15)  # ~15% fraud rate
    n_legit = n_samples - n_fraud

    def make_legit(n: int) -> dict[str, Any]:
        return {
            "amount_claimed": rng.normal(1500, 800, n).clip(100, 10000),
            "num_procedures": rng.integers(1, 5, n),
            "claim_frequency_provider": rng.integers(1, 10, n),
            "days_between_claims": rng.integers(30, 365, n),
            "is_weekend_claim": rng.integers(0, 2, n),
            "amount_vs_avg_ratio": rng.normal(1.0, 0.3, n).clip(0.1, 3.0),
            "num_diagnosis_codes": rng.integers(1, 4, n),
            "procedure_complexity_score": rng.normal(2.0, 0.8, n).clip(0.5, 5.0),
            "is_fraud": np.zeros(n, dtype=int),
        }

    def make_fraud(n: int) -> dict[str, Any]:
        return {
            "amount_claimed": rng.normal(6000, 2000, n).clip(500, 50000),
            "num_procedures": rng.integers(5, 20, n),
            "claim_frequency_provider": rng.integers(15, 50, n),
            "days_between_claims": rng.integers(1, 10, n),
            "is_weekend_claim": rng.integers(0, 2, n),
            "amount_vs_avg_ratio": rng.normal(3.5, 1.0, n).clip(2.0, 10.0),
            "num_diagnosis_codes": rng.integers(5, 15, n),
            "procedure_complexity_score": rng.normal(4.5, 0.5, n).clip(3.0, 5.0),
            "is_fraud": np.ones(n, dtype=int),
        }

    legit_df = pd.DataFrame(make_legit(n_legit))
    fraud_df = pd.DataFrame(make_fraud(n_fraud))
    df = pd.concat([legit_df, fraud_df], ignore_index=True).sample(
        frac=1, random_state=42
    )
    return df


class FraudModelTool:
    """
    ML-powered fraud detection tool using an XGBoost classifier.

    Loads a pre-trained model from disk or trains on synthetic data
    when no model is available.
    """

    def __init__(self, model_path: str | None = None) -> None:
        """
        Initialize FraudModelTool.

        Args:
            model_path: Path to persist/load the model. Defaults to data/fraud_model.joblib.
        """
        self._model_path = Path(model_path or _DEFAULT_MODEL_PATH).resolve()
        self._pipeline = None  # sklearn Pipeline
        self._feature_importances: dict[str, float] = {}
        logger.info("FraudModelTool initialized, model_path='%s'", self._model_path)
        self._ensure_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        """Load model from disk or train a new one if none exists."""
        if self._model_path.exists():
            try:
                self.load_model(str(self._model_path))
                return
            except Exception as exc:
                logger.warning("Failed to load model from disk: %s. Will retrain.", exc)

        logger.info("No pre-trained model found; training on synthetic data...")
        training_data = _generate_synthetic_training_data()
        self.train_model(training_data)
        os.makedirs(self._model_path.parent, exist_ok=True)
        self.save_model(str(self._model_path))

    def _build_pipeline(self):
        """Build the sklearn Pipeline with preprocessing and XGBoost classifier."""
        try:
            from sklearn.pipeline import Pipeline  # noqa: PLC0415
            from sklearn.preprocessing import StandardScaler  # noqa: PLC0415
            from xgboost import XGBClassifier  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "scikit-learn and xgboost are required. "
                "Run: pip install scikit-learn xgboost"
            ) from exc

        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=4,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        return pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, claim_data: dict[str, Any]) -> dict[str, float]:
        """
        Extract ML features from raw claim data dict.

        Args:
            claim_data: Raw claim dictionary with fields like amount_claimed,
                        procedure_codes, diagnosis_codes, claim_date, etc.

        Returns:
            Dict mapping feature names to float values.
        """
        import datetime  # noqa: PLC0415

        amount_claimed = float(claim_data.get("amount_claimed", 0.0))
        procedure_codes: list = claim_data.get("procedure_codes", [])
        diagnosis_codes: list = claim_data.get("diagnosis_codes", [])
        num_procedures = len(procedure_codes)
        num_diagnosis_codes = len(diagnosis_codes)

        # Provider claim frequency: use provider_id if available
        claim_frequency_provider = float(
            claim_data.get("claim_frequency_provider", 1)
        )

        # Days between claims: use a default of 30 if not provided
        days_between_claims = float(claim_data.get("days_between_claims", 30))

        # Check if claim was made on a weekend
        claim_date_str = claim_data.get("claim_date", "")
        is_weekend_claim = 0
        if claim_date_str:
            try:
                if isinstance(claim_date_str, str):
                    claim_date = datetime.date.fromisoformat(claim_date_str[:10])
                else:
                    claim_date = claim_date_str
                is_weekend_claim = int(claim_date.weekday() >= 5)
            except (ValueError, AttributeError):
                is_weekend_claim = 0

        # Amount vs. historical average ratio
        avg_amount = float(claim_data.get("historical_avg_amount", 1500.0))
        amount_vs_avg_ratio = amount_claimed / max(avg_amount, 1.0)

        # Procedure complexity score: simple heuristic (avg code numeric value / 10000)
        complexity_score = 2.0
        if procedure_codes:
            numeric_codes = []
            for code in procedure_codes:
                try:
                    numeric_codes.append(float(str(code).replace("-", "").strip()[:5]))
                except (ValueError, TypeError):
                    pass
            if numeric_codes:
                complexity_score = min(5.0, np.mean(numeric_codes) / 20000.0 * 5.0)

        return {
            "amount_claimed": amount_claimed,
            "num_procedures": float(num_procedures),
            "claim_frequency_provider": claim_frequency_provider,
            "days_between_claims": days_between_claims,
            "is_weekend_claim": float(is_weekend_claim),
            "amount_vs_avg_ratio": amount_vs_avg_ratio,
            "num_diagnosis_codes": float(num_diagnosis_codes),
            "procedure_complexity_score": complexity_score,
        }

    def train_model(self, training_data: pd.DataFrame) -> None:
        """
        Train the XGBoost fraud detection model.

        Args:
            training_data: DataFrame with feature columns and 'is_fraud' label.
        """
        logger.info("Training fraud model on %d samples", len(training_data))
        try:
            X = training_data[FEATURE_NAMES].values
            y = training_data["is_fraud"].values
            pipeline = self._build_pipeline()
            pipeline.fit(X, y)
            self._pipeline = pipeline

            # Store feature importances
            clf = pipeline.named_steps["classifier"]
            importances = clf.feature_importances_
            self._feature_importances = {
                name: float(imp) for name, imp in zip(FEATURE_NAMES, importances)
            }
            logger.info(
                "Model trained successfully. Feature importances: %s",
                self._feature_importances,
            )
        except Exception as exc:
            logger.error("Error training fraud model: %s", exc, exc_info=True)
            raise

    def predict_fraud(self, features: dict[str, Any]) -> dict[str, Any]:
        """
        Predict fraud probability for a given set of features.

        Args:
            features: Dict of feature name -> value. Can be raw claim data
                      (extract_features is called automatically if needed).

        Returns:
            Dict with keys: fraud_score (float 0-1), risk_level (str),
            feature_importance (dict), is_fraud (bool).
        """
        if self._pipeline is None:
            self._ensure_model()

        # Accept either extracted features or raw claim data
        if "amount_claimed" not in features or "num_procedures" not in features:
            features = self.extract_features(features)

        try:
            feature_vector = np.array(
                [[features.get(name, 0.0) for name in FEATURE_NAMES]]
            )
            fraud_proba = float(self._pipeline.predict_proba(feature_vector)[0][1])

            if fraud_proba >= 0.75:
                risk_level = "HIGH"
            elif fraud_proba >= 0.45:
                risk_level = "MEDIUM"
            elif fraud_proba >= 0.25:
                risk_level = "LOW"
            else:
                risk_level = "MINIMAL"

            return {
                "fraud_score": round(fraud_proba, 4),
                "risk_level": risk_level,
                "is_fraud": fraud_proba >= 0.70,
                "feature_importance": self._feature_importances,
                "features_used": {k: float(features.get(k, 0.0)) for k in FEATURE_NAMES},
            }
        except Exception as exc:
            logger.error("Error predicting fraud: %s", exc, exc_info=True)
            return {
                "fraud_score": 0.0,
                "risk_level": "UNKNOWN",
                "is_fraud": False,
                "feature_importance": {},
                "error": str(exc),
            }

    def save_model(self, path: str) -> None:
        """
        Persist the trained model pipeline to disk.

        Args:
            path: File path to save the model (joblib format).
        """
        import joblib  # noqa: PLC0415

        if self._pipeline is None:
            raise RuntimeError("No model to save. Train the model first.")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        joblib.dump(
            {"pipeline": self._pipeline, "feature_importances": self._feature_importances},
            path,
        )
        logger.info("Fraud model saved to '%s'", path)

    def load_model(self, path: str) -> None:
        """
        Load a previously saved model pipeline from disk.

        Args:
            path: File path of the saved model.
        """
        import joblib  # noqa: PLC0415

        data = joblib.load(path)
        self._pipeline = data["pipeline"]
        self._feature_importances = data.get("feature_importances", {})
        logger.info("Fraud model loaded from '%s'", path)
