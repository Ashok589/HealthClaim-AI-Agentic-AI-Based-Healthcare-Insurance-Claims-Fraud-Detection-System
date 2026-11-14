"""
Decision Agent
==============
Final orchestrator that receives results from all upstream agents
(intake, document, policy, medical, fraud) and uses Google Gemini with
chain-of-thought reasoning to produce a definitive claim decision.
"""

import json
import re
from datetime import datetime
from typing import Any, Optional

import google.generativeai as genai
from pydantic import BaseModel, Field

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------

class FinalDecision(BaseModel):
    """Structured result returned by DecisionAgent.make_decision()."""
    status: str                    # APPROVED | REJECTED | UNDER_REVIEW | FRAUD_SUSPECTED
    approved_amount: float = 0.0
    decision_reason: str = ""
    detailed_reasoning: str = ""
    required_actions: list[str] = Field(default_factory=list)
    confidence: float = 0.0        # 0–1 decision confidence
    processing_time_ms: Optional[float] = None
    decision_timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    agent_summary: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

VALID_STATUSES = {"APPROVED", "REJECTED", "UNDER_REVIEW", "FRAUD_SUSPECTED"}


class DecisionAgent:
    """
    AI orchestrator agent that synthesises all upstream agent results into
    a final, explainable claim decision using Gemini chain-of-thought reasoning.
    """

    _SYSTEM_PROMPT = (
        "You are the final decision-making AI for a health insurance claims system. "
        "You receive outputs from four specialist agents:\n"
        "  1. Policy Agent: coverage eligibility and covered amount.\n"
        "  2. Medical Agent: medical necessity and ICD-10/CPT code validity.\n"
        "  3. Fraud Agent: ML fraud score and detected fraud indicators.\n"
        "  4. Document Agent: completeness and authenticity of submitted documents.\n\n"
        "Using chain-of-thought reasoning, determine the final claim decision.\n\n"
        "Decision rules:\n"
        "  - FRAUD_SUSPECTED: fraud_score >= 0.70 OR risk_level in [HIGH, CRITICAL]\n"
        "  - REJECTED: policy not covered, OR not medically necessary (with no flags)\n"
        "  - UNDER_REVIEW: document not authentic, OR any red flags, OR fraud_score 0.40–0.69\n"
        "  - APPROVED: covered + medically necessary + fraud_score < 0.40 + documents OK\n\n"
        "Return ONLY a JSON object with keys:\n"
        "  status (APPROVED|REJECTED|UNDER_REVIEW|FRAUD_SUSPECTED),\n"
        "  approved_amount (number; 0 if not approved),\n"
        "  decision_reason (one concise sentence),\n"
        "  detailed_reasoning (multi-sentence chain-of-thought),\n"
        "  required_actions (array; non-empty only for UNDER_REVIEW),\n"
        "  confidence (0.0–1.0).\n"
        "No markdown fences, no extra text."
    )

    def __init__(self) -> None:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL)
        logger.info("DecisionAgent initialised with model=%s", settings.LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make_decision(self, all_agent_results: dict) -> FinalDecision:
        """
        Make a final claim approval decision.

        Parameters
        ----------
        all_agent_results : dict
            Expected keys: 'claim_intake', 'document', 'policy', 'medical', 'fraud'.
            Values are the result dicts/models from each respective agent.

        Returns
        -------
        FinalDecision
        """
        t_start = datetime.utcnow()
        logger.info("DecisionAgent.make_decision() called with agent keys: %s",
                    list(all_agent_results.keys()))

        try:
            # Build a compact summary of agent results for the AI prompt
            summary = self._build_summary(all_agent_results)

            # Apply hard pre-checks before calling Gemini (saves tokens & latency)
            pre_check = self._pre_check_hard_rules(summary)
            if pre_check:
                elapsed = (datetime.utcnow() - t_start).total_seconds() * 1000
                pre_check.processing_time_ms = elapsed
                pre_check.agent_summary = summary
                return pre_check

            # Use Gemini for chain-of-thought decision
            ai_decision = self._ai_decide(summary)
            status = ai_decision.get("status", "UNDER_REVIEW").upper()
            if status not in VALID_STATUSES:
                status = "UNDER_REVIEW"

            # Determine approved amount
            approved_amount = 0.0
            if status == "APPROVED":
                policy_result = all_agent_results.get("policy") or {}
                covered = self._get_nested(policy_result, "covered_amount", 0.0)
                approved_amount = float(
                    ai_decision.get("approved_amount") or covered or 0.0
                )

            elapsed = (datetime.utcnow() - t_start).total_seconds() * 1000
            return FinalDecision(
                status=status,
                approved_amount=round(approved_amount, 2),
                decision_reason=ai_decision.get("decision_reason", ""),
                detailed_reasoning=ai_decision.get("detailed_reasoning", ""),
                required_actions=ai_decision.get("required_actions", []),
                confidence=float(ai_decision.get("confidence", 0.0)),
                processing_time_ms=round(elapsed, 2),
                agent_summary=summary,
            )

        except Exception as exc:
            logger.error("DecisionAgent.make_decision error: %s", exc, exc_info=True)
            elapsed = (datetime.utcnow() - t_start).total_seconds() * 1000
            return FinalDecision(
                status="UNDER_REVIEW",
                decision_reason="Decision engine error — manual review required.",
                detailed_reasoning=str(exc),
                required_actions=["Manual adjuster review required due to system error."],
                processing_time_ms=round(elapsed, 2),
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_summary(self, results: dict) -> dict:
        """Extract key signals from each agent result for the AI prompt."""
        def _safe(d, *keys, default=None):
            for k in keys:
                if isinstance(d, dict):
                    d = d.get(k, default)
                elif hasattr(d, k):
                    d = getattr(d, k, default)
                else:
                    return default
            return d

        policy = results.get("policy") or {}
        medical = results.get("medical") or {}
        fraud = results.get("fraud") or {}
        document = results.get("document") or {}
        intake = results.get("claim_intake") or {}

        return {
            "policy": {
                "is_covered": _safe(policy, "is_covered", default=False),
                "covered_amount": _safe(policy, "covered_amount", default=0.0),
                "exclusions": _safe(policy, "exclusions", default=[]),
                "reason": _safe(policy, "reason", default=""),
                "coverage_warnings": _safe(policy, "coverage_warnings", default=[]),
            },
            "medical": {
                "is_medically_necessary": _safe(medical, "is_medically_necessary", default=False),
                "icd10_valid": _safe(medical, "icd10_valid", default=False),
                "cpt_valid": _safe(medical, "cpt_valid", default=False),
                "red_flags": _safe(medical, "red_flags", default=[]),
                "recommendations": _safe(medical, "recommendations", default=""),
            },
            "fraud": {
                "fraud_score": _safe(fraud, "fraud_score", default=0.0),
                "risk_level": _safe(fraud, "risk_level", default="LOW"),
                "fraud_flags": _safe(fraud, "fraud_flags", default=[]),
                "recommendation": _safe(fraud, "recommendation", default=""),
            },
            "document": {
                "is_authentic": _safe(document, "is_authentic", default=False),
                "success": _safe(document, "success", default=False),
                "confidence_scores": _safe(document, "confidence_scores", default={}),
            },
            "intake": {
                "success": _safe(intake, "success", default=False),
                "amount_claimed": _safe(intake, "amount_claimed", default=0.0),
                "validation_errors": _safe(intake, "validation_errors", default=[]),
            },
        }

    def _pre_check_hard_rules(self, summary: dict) -> Optional[FinalDecision]:
        """Apply deterministic hard rules before calling Gemini."""
        # Critical fraud
        fraud_score = float(summary["fraud"]["fraud_score"])
        risk_level = str(summary["fraud"]["risk_level"]).upper()
        if fraud_score >= float(getattr(settings, "FRAUD_THRESHOLD", 0.7)) or risk_level in {"HIGH", "CRITICAL"}:
            return FinalDecision(
                status="FRAUD_SUSPECTED",
                approved_amount=0.0,
                decision_reason="High fraud score detected by ML model.",
                detailed_reasoning=(
                    f"Fraud score={fraud_score:.3f}, risk_level={risk_level}. "
                    f"Flags: {summary['fraud']['fraud_flags']}"
                ),
                required_actions=["Refer to fraud investigation team.", "Freeze claim payment."],
                confidence=0.95,
            )

        # Intake failed completely
        if not summary["intake"]["success"]:
            return FinalDecision(
                status="REJECTED",
                decision_reason="Claim intake validation failed.",
                detailed_reasoning=f"Errors: {summary['intake']['validation_errors']}",
                confidence=0.99,
            )

        return None  # No hard rule triggered; proceed to AI

    def _ai_decide(self, summary: dict) -> dict:
        """Call Gemini with chain-of-thought to produce a JSON decision."""
        prompt = (
            f"{self._SYSTEM_PROMPT}\n\n"
            f"Agent results summary:\n{json.dumps(summary, indent=2, default=str)}"
        )
        response = self._model.generate_content(prompt)
        text = response.text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        return json.loads(text)

    @staticmethod
    def _get_nested(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)
