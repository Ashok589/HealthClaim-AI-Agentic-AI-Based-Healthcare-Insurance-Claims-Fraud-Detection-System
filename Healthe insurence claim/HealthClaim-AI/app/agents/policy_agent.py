"""
Policy Agent
============
Checks whether a claim is covered under the patient's insurance policy,
calculates the covered amount after deductibles/copay, and returns a
detailed PolicyCheckResult.  Uses RAG (via app.rag.retriever) to search
the policy database for relevant policy documents.
"""

import json
import re
from datetime import date
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

class PolicyCheckResult(BaseModel):
    """Structured result returned by PolicyAgent.check_coverage()."""
    is_covered: bool
    covered_amount: float = 0.0
    deductible_applied: float = 0.0
    copay_applied: float = 0.0
    reason: str = ""
    exclusions: list[str] = Field(default_factory=list)
    policy_details: dict[str, Any] = Field(default_factory=dict)
    uncovered_procedures: list[str] = Field(default_factory=list)
    coverage_warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class PolicyAgent:
    """
    AI agent that validates insurance coverage for a submitted claim.

    Uses RAG-based retrieval to find the relevant policy document and
    Gemini to reason about coverage eligibility.
    """

    _COVERAGE_PROMPT = (
        "You are an expert insurance policy analyst. "
        "Given the policy details and claim information below, determine:\n"
        "1. Is the claim covered? (yes/no)\n"
        "2. Which specific procedures are NOT covered and why?\n"
        "3. Are there any exclusions that apply?\n"
        "4. Any warnings or concerns about coverage?\n\n"
        "Respond with a JSON object using keys:\n"
        "  is_covered (bool), uncovered_procedures (array of strings),\n"
        "  exclusions (array of strings), warnings (array of strings),\n"
        "  reasoning (string).\n"
        "Return ONLY the JSON, no markdown fences."
    )

    def __init__(self) -> None:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self._model = genai.GenerativeModel(settings.LLM_MODEL)
        self._retriever = self._init_retriever()
        logger.info("PolicyAgent initialised with model=%s", settings.LLM_MODEL)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_coverage(self, claim_data: dict, policy_id: str) -> PolicyCheckResult:
        """
        Check whether claim_data is covered under the given policy_id.

        Parameters
        ----------
        claim_data : dict
            Normalised claim data (from ClaimIntakeAgent or the API layer).
        policy_id : str
            The policy identifier to look up.

        Returns
        -------
        PolicyCheckResult
        """
        logger.info("Checking coverage for policy_id=%s", policy_id)
        try:
            # 1. Retrieve policy document
            policy = self._fetch_policy(policy_id)
            if policy is None:
                return PolicyCheckResult(
                    is_covered=False,
                    reason=f"Policy '{policy_id}' not found.",
                    error="Policy not found",
                )

            # 2. Check policy active dates
            date_warning = self._check_policy_dates(policy, claim_data.get("claim_date"))

            # 3. AI-assisted coverage analysis
            ai_result = self._ai_coverage_check(claim_data, policy)

            # 4. Financial calculation
            amount_claimed = float(claim_data.get("amount_claimed", 0))
            covered_amount, deductible_applied, copay_applied = self._calculate_amounts(
                amount_claimed, policy, ai_result.get("is_covered", False)
            )

            warnings = ai_result.get("warnings", [])
            if date_warning:
                warnings.append(date_warning)

            return PolicyCheckResult(
                is_covered=ai_result.get("is_covered", False),
                covered_amount=covered_amount,
                deductible_applied=deductible_applied,
                copay_applied=copay_applied,
                reason=ai_result.get("reasoning", ""),
                exclusions=ai_result.get("exclusions", []),
                uncovered_procedures=ai_result.get("uncovered_procedures", []),
                coverage_warnings=warnings,
                policy_details={
                    "policy_number": policy.get("policy_number"),
                    "coverage_type": policy.get("coverage_type"),
                    "holder_name": policy.get("holder_name"),
                    "max_coverage": policy.get("max_coverage"),
                    "deductible": policy.get("deductible"),
                    "copay_percent": policy.get("copay_percent"),
                    "effective_date": str(policy.get("effective_date", "")),
                    "expiry_date": str(policy.get("expiry_date", "")),
                },
            )

        except Exception as exc:
            logger.error("PolicyAgent.check_coverage error: %s", exc, exc_info=True)
            return PolicyCheckResult(
                is_covered=False,
                reason=f"Coverage check failed: {exc}",
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_retriever(self):
        """Attempt to initialise the RAG retriever; fall back gracefully."""
        try:
            from app.rag.retriever import PolicyRetriever  # type: ignore
            return PolicyRetriever()
        except Exception as exc:
            logger.warning("RAG retriever unavailable (%s) – using JSON file fallback.", exc)
            return None

    def _fetch_policy(self, policy_id: str) -> Optional[dict]:
        """Fetch a policy document via RAG retriever or JSON file fallback."""
        if self._retriever:
            try:
                docs = self._retriever.search(policy_id, top_k=1)
                if docs:
                    return docs[0]
            except Exception as exc:
                logger.warning("RAG search failed: %s", exc)

        # Fallback: load from sample JSON file
        try:
            sample_path = (
                "data/policies/sample_policy.json"
            )
            with open(sample_path, encoding="utf-8") as fh:
                policies: list[dict] = json.load(fh)
            for p in policies:
                if p.get("policy_number") == policy_id or p.get("id") == policy_id:
                    return p
            # If not found by ID, return first policy for demo
            if policies:
                logger.warning("Policy %s not found; using first sample policy.", policy_id)
                return policies[0]
        except Exception as exc:
            logger.warning("Sample policy load failed: %s", exc)

        return None

    def _check_policy_dates(self, policy: dict, claim_date: Optional[str]) -> Optional[str]:
        """Return a warning string if the policy is expired or not yet active."""
        try:
            today = date.fromisoformat(claim_date) if claim_date else date.today()
            eff = date.fromisoformat(str(policy.get("effective_date", "2000-01-01")))
            exp = date.fromisoformat(str(policy.get("expiry_date", "2099-12-31")))
            if today < eff:
                return f"Policy not yet active on {claim_date} (effective {eff})"
            if today > exp:
                return f"Policy expired on {exp} (claim date {claim_date})"
        except Exception:
            pass
        return None

    def _ai_coverage_check(self, claim_data: dict, policy: dict) -> dict:
        """Call Gemini to reason about coverage eligibility."""
        prompt = (
            f"{self._COVERAGE_PROMPT}\n\n"
            f"Policy:\n{json.dumps(policy, indent=2, default=str)}\n\n"
            f"Claim:\n{json.dumps(claim_data, indent=2, default=str)}"
        )
        try:
            response = self._model.generate_content(prompt)
            text = response.text.strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            return json.loads(text)
        except Exception as exc:
            logger.error("AI coverage check failed: %s", exc)
            return {
                "is_covered": False,
                "reasoning": f"AI analysis unavailable: {exc}",
                "exclusions": [],
                "uncovered_procedures": [],
                "warnings": ["AI analysis failed; manual review required."],
            }

    @staticmethod
    def _calculate_amounts(
        amount_claimed: float, policy: dict, is_covered: bool
    ) -> tuple[float, float, float]:
        """
        Calculate covered_amount, deductible_applied, copay_applied.

        Returns (covered_amount, deductible_applied, copay_applied).
        """
        if not is_covered or amount_claimed <= 0:
            return 0.0, 0.0, 0.0

        deductible = float(policy.get("deductible", 0))
        copay_pct = float(policy.get("copay_percent", 0)) / 100.0
        max_coverage = float(policy.get("max_coverage", float("inf")))

        after_deductible = max(0.0, amount_claimed - deductible)
        copay_amount = after_deductible * copay_pct
        covered = min(after_deductible - copay_amount, max_coverage)
        covered = round(max(0.0, covered), 2)
        return covered, round(deductible, 2), round(copay_amount, 2)
