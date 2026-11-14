"""
Fraud Agent
===========
Detects potentially fraudulent insurance claims using a combination of a
scikit-learn / XGBoost ML model (trained on-the-fly when no saved model is
found) and Google Gemini's reasoning capabilities.

Fraud indicators checked:
  - Upcoding (procedure code billed at higher complexity than justified)
  - Unbundling (billing separately for procedures normally bundled)
  - Phantom billing (procedures not supported by clinical notes)
  - Duplicate claims (same patient/provider/date/amount in history)
  - Unusual claim timing patterns
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import google.generativeai as genai
from pydantic import BaseModel, Field

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

MODEL_PATH = Path("app/models/fraud_model.pkl")


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class FraudAnalysisResult(BaseModel):
    """Structured result returned by FraudAgent.analyze_fraud()."""
    fraud_score: float
    fraud_flags: list[str] = Field(default_factory=list)
    risk_level: str = "LOW"       # LOW | MEDIUM | HIGH | CRITICAL
    explanation: str = ""
    recommendation: str = ""
    feature_values: dict[str, float] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class FraudAgent:
    """
    AI + ML agent for insurance claim fraud detection.

    ML model:  Loaded from app/models/fraud_model.pkl when available;
               otherwise a simple RandomForest is trained on synthetic data.
    AI model:  Google Gemini for human-readable fraud pattern analysis.
    """

    _FRAUD_PROMPT = (
        "You are an expert insurance fraud investigator. "
        "Analyze the following claim data and ML fraud score for potential fraud.\n\n"
        "Identify specific fraud types if present:\n"
        "  - Upcoding: billing higher-complexity codes than warranted\n"
        "  - Unbundling: splitting bundled procedures into separate charges\n"
        "  - Phantom billing: charging for services not rendered\n"
        "  - Duplicate claims: same procedure/patient billed multiple times\n"
        "  - Unusual timing: claims submitted outside normal patterns\n\n"
        "Return ONLY a JSON object with keys:\n"
        "  fraud_flags (array of strings), explanation (string),\n"
        "  recommendation (string).\n"
        "No markdown fences."
    )

    def __init__(self) -> None:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL)
        self._ml_model = self._load_or_train_ml_model()
        logger.info("FraudAgent initialised with model=%s", settings.LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_fraud(
        self, claim_data: dict, historical_data: Optional[list[dict]] = None
    ) -> FraudAnalysisResult:
        """
        Assess fraud probability for a claim.

        Parameters
        ----------
        claim_data : dict
            Normalised claim data dict.
        historical_data : list[dict] | None
            Prior claims for the same patient/provider (for pattern detection).

        Returns
        -------
        FraudAnalysisResult
        """
        logger.info(
            "Analyzing fraud for patient=%s amount=%.2f",
            claim_data.get("patient_name", "?"),
            float(claim_data.get("amount_claimed", 0)),
        )
        try:
            # 1. Feature engineering
            features = self._extract_features(claim_data, historical_data or [])

            # 2. ML fraud score
            fraud_score = self._ml_predict(features)

            # 3. Rule-based flags (fast, deterministic)
            rule_flags = self._rule_based_checks(claim_data, historical_data or [])

            # 4. Gemini narrative analysis
            ai_result = self._ai_fraud_analysis(claim_data, fraud_score, rule_flags)

            # 5. Merge flags & determine risk level
            all_flags = list(set(rule_flags + ai_result.get("fraud_flags", [])))
            risk_level = self._score_to_risk(fraud_score, len(all_flags))

            return FraudAnalysisResult(
                fraud_score=round(fraud_score, 4),
                fraud_flags=all_flags,
                risk_level=risk_level,
                explanation=ai_result.get("explanation", ""),
                recommendation=ai_result.get("recommendation", ""),
                feature_values={k: round(float(v), 4) for k, v in features.items()},
            )

        except Exception as exc:
            logger.error("FraudAgent.analyze_fraud error: %s", exc, exc_info=True)
            return FraudAnalysisResult(
                fraud_score=0.0,
                risk_level="UNKNOWN",
                explanation=f"Fraud analysis failed: {exc}",
                recommendation="Manual review required.",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _extract_features(self, claim: dict, history: list[dict]) -> dict[str, float]:
        """Build a numeric feature vector from claim and history."""
        amount = float(claim.get("amount_claimed", 0))
        diagnosis_codes = claim.get("diagnosis_codes", [])
        procedure_codes = claim.get("procedure_codes", [])
        num_procedures = len(procedure_codes)
        num_diagnoses = len(diagnosis_codes)

        # Days since last claim from same provider
        days_since_last = self._days_since_last_claim(claim, history)

        # Provider claim frequency (claims per month in history)
        provider_id = claim.get("provider_id", "")
        provider_freq = sum(
            1 for h in history if h.get("provider_id") == provider_id
        )

        # Unusual code flag: very high num_procedures with low num_diagnoses
        code_ratio = num_procedures / max(num_diagnoses, 1)

        # Round-number amount suspicion
        is_round_amount = 1.0 if amount % 100 == 0 else 0.0

        # High-amount flag (> $10k)
        is_high_amount = 1.0 if amount > 10_000 else 0.0

        return {
            "amount_claimed": amount,
            "num_procedures": float(num_procedures),
            "num_diagnoses": float(num_diagnoses),
            "days_since_last_claim": float(days_since_last),
            "provider_claim_frequency": float(provider_freq),
            "code_ratio": float(code_ratio),
            "is_round_amount": is_round_amount,
            "is_high_amount": is_high_amount,
        }

    # ------------------------------------------------------------------
    # ML helpers
    # ------------------------------------------------------------------

    def _load_or_train_ml_model(self):
        """Load a pre-trained model or train a simple RandomForest on synthetic data."""
        import joblib  # type: ignore

        if MODEL_PATH.exists():
            try:
                model = joblib.load(str(MODEL_PATH))
                logger.info("Fraud model loaded from %s", MODEL_PATH)
                return model
            except Exception as exc:
                logger.warning("Could not load fraud model (%s); training new one.", exc)

        logger.info("Training synthetic fraud detection model …")
        return self._train_synthetic_model()

    def _train_synthetic_model(self):
        """Train a RandomForest on small synthetic data and persist it."""
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        import joblib  # type: ignore

        rng = np.random.RandomState(42)
        n = 500
        # Feature order matches _extract_features keys
        X = np.column_stack([
            rng.exponential(2000, n),          # amount_claimed
            rng.randint(1, 10, n).astype(float),  # num_procedures
            rng.randint(1, 6, n).astype(float),   # num_diagnoses
            rng.randint(0, 365, n).astype(float),  # days_since_last
            rng.randint(0, 50, n).astype(float),   # provider_freq
            rng.uniform(0.5, 5, n),            # code_ratio
            rng.randint(0, 2, n).astype(float),    # is_round_amount
            (rng.exponential(2000, n) > 10000).astype(float),  # is_high_amount
        ])
        # Synthetic labels: fraud if high amount + low days + high freq
        y = (
            (X[:, 0] > 8000) & (X[:, 3] < 30) & (X[:, 4] > 20)
        ).astype(int)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        import joblib  # type: ignore
        joblib.dump(clf, str(MODEL_PATH))
        logger.info("Synthetic fraud model trained and saved to %s", MODEL_PATH)
        return clf

    def _ml_predict(self, features: dict[str, float]) -> float:
        """Run ML model and return fraud probability (0–1)."""
        feature_order = [
            "amount_claimed", "num_procedures", "num_diagnoses",
            "days_since_last_claim", "provider_claim_frequency",
            "code_ratio", "is_round_amount", "is_high_amount",
        ]
        X = np.array([[features.get(k, 0.0) for k in feature_order]])
        try:
            prob = self._ml_model.predict_proba(X)[0][1]
            return float(prob)
        except Exception:
            return float(self._ml_model.predict(X)[0])

    # ------------------------------------------------------------------
    # Rule-based & AI helpers
    # ------------------------------------------------------------------

    def _rule_based_checks(self, claim: dict, history: list[dict]) -> list[str]:
        """Fast heuristic fraud flag detection."""
        flags: list[str] = []
        amount = float(claim.get("amount_claimed", 0))
        proc = claim.get("procedure_codes", [])
        diag = claim.get("diagnosis_codes", [])

        # Upcoding heuristic: too many high-complexity codes
        high_complexity = [c for c in proc if c.startswith(("99215", "99214", "99213"))]
        if len(high_complexity) > 2:
            flags.append("UPCODING: Multiple high-complexity E&M codes billed simultaneously")

        # Unbundling: excessive number of procedure codes for claimed amount
        if len(proc) > 10 and amount < 1000:
            flags.append("UNBUNDLING: High procedure count relative to claimed amount")

        # Duplicate detection: same patient + provider + amount in history
        pid = claim.get("patient_id")
        prov = claim.get("provider_id")
        for h in history:
            if (
                h.get("patient_id") == pid
                and h.get("provider_id") == prov
                and abs(float(h.get("amount_claimed", -1)) - amount) < 1.0
            ):
                flags.append("DUPLICATE_CLAIM: Identical claim found in history")
                break

        # Phantom billing: procedures with no matching diagnoses context
        if proc and not diag:
            flags.append("PHANTOM_BILLING: Procedures claimed with no diagnosis codes")

        # Unusual timing: very recent repeated claims
        days = self._days_since_last_claim(claim, history)
        if 0 < days < 3:
            flags.append("UNUSUAL_TIMING: Multiple claims within 3 days from same provider")

        return flags

    def _ai_fraud_analysis(
        self, claim: dict, fraud_score: float, rule_flags: list[str]
    ) -> dict:
        """Use Gemini to generate a narrative fraud analysis."""
        payload = {
            "claim_summary": {
                "patient_id": claim.get("patient_id"),
                "provider_id": claim.get("provider_id"),
                "amount_claimed": claim.get("amount_claimed"),
                "diagnosis_codes": claim.get("diagnosis_codes"),
                "procedure_codes": claim.get("procedure_codes"),
            },
            "ml_fraud_score": fraud_score,
            "rule_based_flags": rule_flags,
        }
        prompt = (
            f"{self._FRAUD_PROMPT}\n\nClaim data:\n{json.dumps(payload, indent=2)}"
        )
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except Exception as exc:
            logger.warning("Gemini fraud analysis failed: %s", exc)
            return {
                "fraud_flags": [],
                "explanation": f"AI analysis unavailable: {exc}",
                "recommendation": "Manual review recommended.",
            }

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _days_since_last_claim(claim: dict, history: list[dict]) -> float:
        """Return days between this claim and the most recent historical claim."""
        claim_date_str = claim.get("claim_date") or datetime.utcnow().strftime("%Y-%m-%d")
        try:
            claim_date = datetime.fromisoformat(claim_date_str)
        except ValueError:
            return 999.0

        pid = claim.get("patient_id")
        prov = claim.get("provider_id")
        dates = []
        for h in history:
            if h.get("patient_id") == pid or h.get("provider_id") == prov:
                try:
                    d = datetime.fromisoformat(str(h.get("claim_date", "")))
                    dates.append(d)
                except ValueError:
                    pass
        if not dates:
            return 999.0
        most_recent = max(dates)
        delta = abs((claim_date - most_recent).days)
        return float(delta)

    @staticmethod
    def _score_to_risk(score: float, num_flags: int) -> str:
        """Map fraud score and flag count to a risk level string."""
        adjusted = score + num_flags * 0.05
        if adjusted >= 0.75:
            return "CRITICAL"
        elif adjusted >= 0.55:
            return "HIGH"
        elif adjusted >= 0.35:
            return "MEDIUM"
        return "LOW"
