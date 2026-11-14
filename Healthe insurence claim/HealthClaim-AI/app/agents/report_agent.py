"""
Report Agent
============
Generates a comprehensive PDF report and a JSON summary for a processed
insurance claim, including all intermediate agent results, the final
decision, and a processing timeline.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.utils.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

REPORTS_DIR = Path("data/reports")


# ---------------------------------------------------------------------------
# Agent class
# ---------------------------------------------------------------------------

class ReportAgent:
    """
    AI report-generation agent that produces:
      - A comprehensive, formatted PDF using ReportLab.
      - A structured JSON summary dict suitable for API responses.
    """

    def __init__(self) -> None:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("ReportAgent initialised; reports directory=%s", REPORTS_DIR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pdf_report(
        self, claim_data: dict, decision_data: dict
    ) -> str:
        """
        Generate a full PDF report for the processed claim.

        Parameters
        ----------
        claim_data : dict
            Combined claim information including patient, provider, codes, amounts,
            and intermediate agent results (keys: 'policy', 'medical', 'fraud', 'document').
        decision_data : dict
            Final decision dict (from DecisionAgent) with status, approved_amount, reasoning.

        Returns
        -------
        str
            Absolute file path to the generated PDF.
        """
        try:
            from reportlab.lib.pagesizes import A4  # type: ignore
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # type: ignore
            from reportlab.lib.units import cm  # type: ignore
            from reportlab.lib import colors  # type: ignore
            from reportlab.platypus import (  # type: ignore
                SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                HRFlowable, KeepTogether,
            )

            claim_number = claim_data.get("claim_number") or claim_data.get(
                "claim_id", datetime.utcnow().strftime("%Y%m%d%H%M%S")
            )
            filename = f"claim_report_{claim_number}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.pdf"
            output_path = REPORTS_DIR / filename

            doc = SimpleDocTemplate(
                str(output_path),
                pagesize=A4,
                rightMargin=2 * cm,
                leftMargin=2 * cm,
                topMargin=2 * cm,
                bottomMargin=2 * cm,
            )
            styles = getSampleStyleSheet()
            story = []

            # ---- Title ----
            title_style = ParagraphStyle(
                "Title",
                parent=styles["Title"],
                fontSize=18,
                textColor=colors.HexColor("#1a365d"),
                spaceAfter=12,
            )
            story.append(Paragraph("HealthClaim AI — Claim Processing Report", title_style))
            story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1a365d")))
            story.append(Spacer(1, 0.4 * cm))

            # ---- Claim Summary ----
            story.extend(self._section_header("Claim Summary", styles))
            story.append(self._kv_table([
                ("Claim Number", claim_number),
                ("Generated At", datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")),
                ("Status", decision_data.get("status", "N/A")),
                ("Amount Claimed", f"${float(claim_data.get('amount_claimed', 0)):,.2f}"),
                ("Approved Amount", f"${float(decision_data.get('approved_amount', 0)):,.2f}"),
            ], styles))
            story.append(Spacer(1, 0.3 * cm))

            # ---- Patient / Provider ----
            story.extend(self._section_header("Patient & Provider Information", styles))
            patient = claim_data.get("patient_info") or claim_data
            provider = claim_data.get("provider_info") or claim_data
            story.append(self._kv_table([
                ("Patient Name", self._nested(patient, "patient_name", "N/A")),
                ("Patient ID", self._nested(patient, "patient_id", "N/A")),
                ("Date of Birth", self._nested(patient, "date_of_birth", "N/A")),
                ("Provider Name", self._nested(provider, "provider_name", "N/A")),
                ("Provider ID", self._nested(provider, "provider_id", "N/A")),
                ("Claim Date", claim_data.get("claim_date", "N/A")),
            ], styles))
            story.append(Spacer(1, 0.3 * cm))

            # ---- Medical Codes ----
            story.extend(self._section_header("Medical Codes", styles))
            diag = ", ".join(claim_data.get("diagnosis_codes", [])) or "None"
            proc = ", ".join(claim_data.get("procedure_codes", [])) or "None"
            story.append(self._kv_table([
                ("Diagnosis Codes (ICD-10)", diag),
                ("Procedure Codes (CPT)", proc),
            ], styles))
            story.append(Spacer(1, 0.3 * cm))

            # ---- Policy Coverage ----
            policy = claim_data.get("policy") or decision_data.get("agent_summary", {}).get("policy", {})
            story.extend(self._section_header("Policy Coverage Analysis", styles))
            story.append(self._kv_table([
                ("Is Covered", str(self._nested(policy, "is_covered", "N/A"))),
                ("Covered Amount", f"${float(self._nested(policy, 'covered_amount', 0)):,.2f}"),
                ("Reason", self._nested(policy, "reason", "N/A")),
                ("Exclusions", ", ".join(self._nested(policy, "exclusions", []) or []) or "None"),
            ], styles))
            story.append(Spacer(1, 0.3 * cm))

            # ---- Medical Validation ----
            medical = claim_data.get("medical") or decision_data.get("agent_summary", {}).get("medical", {})
            story.extend(self._section_header("Medical Necessity Validation", styles))
            story.append(self._kv_table([
                ("Medically Necessary", str(self._nested(medical, "is_medically_necessary", "N/A"))),
                ("ICD-10 Valid", str(self._nested(medical, "icd10_valid", "N/A"))),
                ("CPT Valid", str(self._nested(medical, "cpt_valid", "N/A"))),
                ("Red Flags", ", ".join(self._nested(medical, "red_flags", []) or []) or "None"),
                ("Recommendations", self._nested(medical, "recommendations", "N/A")),
            ], styles))
            story.append(Spacer(1, 0.3 * cm))

            # ---- Fraud Assessment ----
            fraud = claim_data.get("fraud") or decision_data.get("agent_summary", {}).get("fraud", {})
            story.extend(self._section_header("Fraud Assessment", styles))
            story.append(self._kv_table([
                ("Fraud Score", f"{float(self._nested(fraud, 'fraud_score', 0)):.4f}"),
                ("Risk Level", str(self._nested(fraud, "risk_level", "N/A"))),
                ("Fraud Flags", ", ".join(self._nested(fraud, "fraud_flags", []) or []) or "None"),
                ("Recommendation", self._nested(fraud, "recommendation", "N/A")),
            ], styles))
            story.append(Spacer(1, 0.3 * cm))

            # ---- Final Decision ----
            story.extend(self._section_header("Final Decision", styles))
            status = decision_data.get("status", "N/A")
            status_color = {
                "APPROVED": colors.green,
                "REJECTED": colors.red,
                "UNDER_REVIEW": colors.orange,
                "FRAUD_SUSPECTED": colors.darkred,
            }.get(status, colors.black)
            status_style = ParagraphStyle(
                "StatusStyle", parent=styles["Normal"],
                textColor=status_color, fontSize=13, fontName="Helvetica-Bold",
            )
            story.append(Paragraph(f"Decision: {status}", status_style))
            story.append(Spacer(1, 0.2 * cm))
            story.append(self._kv_table([
                ("Decision Reason", decision_data.get("decision_reason", "N/A")),
                ("Approved Amount", f"${float(decision_data.get('approved_amount', 0)):,.2f}"),
                ("Confidence", f"{float(decision_data.get('confidence', 0)):.1%}"),
            ], styles))
            story.append(Spacer(1, 0.2 * cm))
            story.append(Paragraph("<b>Detailed Reasoning:</b>", styles["Normal"]))
            story.append(Paragraph(
                decision_data.get("detailed_reasoning", "N/A"), styles["Normal"]
            ))

            # Required actions
            req_actions = decision_data.get("required_actions") or []
            if req_actions:
                story.append(Spacer(1, 0.2 * cm))
                story.append(Paragraph("<b>Required Actions:</b>", styles["Normal"]))
                for action in req_actions:
                    story.append(Paragraph(f"• {action}", styles["Normal"]))

            # ---- Processing Timeline ----
            story.append(Spacer(1, 0.4 * cm))
            story.extend(self._section_header("Processing Timeline", styles))
            timeline_entries = self._build_timeline(claim_data, decision_data)
            for entry in timeline_entries:
                story.append(Paragraph(f"• {entry}", styles["Normal"]))

            # Build PDF
            doc.build(story)
            logger.info("PDF report generated: %s", output_path)
            return str(output_path.resolve())

        except ImportError:
            logger.error("reportlab is not installed – cannot generate PDF.")
            raise RuntimeError("Install reportlab to generate PDF reports: pip install reportlab")
        except Exception as exc:
            logger.error("PDF generation failed: %s", exc, exc_info=True)
            raise

    def generate_json_summary(
        self, claim_data: dict, decision_data: dict
    ) -> dict:
        """
        Generate a structured JSON summary of the claim processing result.

        Parameters
        ----------
        claim_data : dict
            Claim data including agent results.
        decision_data : dict
            Final decision from DecisionAgent.

        Returns
        -------
        dict
            JSON-serialisable summary suitable for API responses.
        """
        claim_number = claim_data.get("claim_number") or claim_data.get("claim_id", "UNKNOWN")
        patient = claim_data.get("patient_info") or {}
        provider = claim_data.get("provider_info") or {}

        summary = {
            "claim_number": claim_number,
            "generated_at": datetime.utcnow().isoformat(),
            "status": decision_data.get("status"),
            "claim": {
                "patient_name": self._nested(patient, "patient_name"),
                "patient_id": self._nested(patient, "patient_id"),
                "provider_name": self._nested(provider, "provider_name"),
                "provider_id": self._nested(provider, "provider_id"),
                "claim_date": claim_data.get("claim_date"),
                "amount_claimed": claim_data.get("amount_claimed"),
                "diagnosis_codes": claim_data.get("diagnosis_codes", []),
                "procedure_codes": claim_data.get("procedure_codes", []),
            },
            "decision": {
                "status": decision_data.get("status"),
                "approved_amount": decision_data.get("approved_amount", 0.0),
                "decision_reason": decision_data.get("decision_reason"),
                "detailed_reasoning": decision_data.get("detailed_reasoning"),
                "confidence": decision_data.get("confidence", 0.0),
                "required_actions": decision_data.get("required_actions", []),
                "decision_timestamp": decision_data.get("decision_timestamp"),
            },
            "agent_results": {
                "policy": claim_data.get("policy") or decision_data.get("agent_summary", {}).get("policy"),
                "medical": claim_data.get("medical") or decision_data.get("agent_summary", {}).get("medical"),
                "fraud": claim_data.get("fraud") or decision_data.get("agent_summary", {}).get("fraud"),
                "document": claim_data.get("document") or decision_data.get("agent_summary", {}).get("document"),
            },
            "processing_time_ms": decision_data.get("processing_time_ms"),
        }

        # Persist JSON alongside PDFs
        json_filename = f"claim_summary_{claim_number}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.json"
        json_path = REPORTS_DIR / json_filename
        try:
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2, default=str)
            logger.info("JSON summary saved: %s", json_path)
        except Exception as exc:
            logger.warning("Could not persist JSON summary: %s", exc)

        return summary

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _nested(obj: Any, key: str, default: Any = None) -> Any:
        """Safely get a key from a dict or attribute from an object."""
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _section_header(title: str, styles) -> list:
        """Return a list of flowables for a styled section header."""
        from reportlab.platypus import Paragraph, Spacer, HRFlowable  # type: ignore
        from reportlab.lib import colors  # type: ignore
        from reportlab.lib.styles import ParagraphStyle  # type: ignore
        from reportlab.lib.units import cm  # type: ignore

        header_style = ParagraphStyle(
            f"Header_{title}",
            parent=styles["Heading2"],
            textColor=colors.HexColor("#2c5282"),
            spaceBefore=8,
            spaceAfter=4,
        )
        return [
            Paragraph(title, header_style),
            HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#bee3f8")),
            Spacer(1, 0.2 * cm),
        ]

    @staticmethod
    def _kv_table(rows: list[tuple[str, str]], styles) -> Any:
        """Build a two-column key-value Table flowable."""
        from reportlab.platypus import Table, TableStyle  # type: ignore
        from reportlab.lib import colors  # type: ignore

        data = [[k, str(v)] for k, v in rows]
        table = Table(data, colWidths=["35%", "65%"])
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#ebf8ff")),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#bee3f8")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 6),
            ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#f7fafc")]),
        ]))
        return table

    @staticmethod
    def _build_timeline(claim_data: dict, decision_data: dict) -> list[str]:
        """Build a text-based processing timeline from available timestamps."""
        entries = []
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        entries.append(f"Claim submitted — {claim_data.get('claim_date', 'N/A')}")
        entries.append("Intake validation completed")
        entries.append("Document extraction completed")
        entries.append("Policy coverage checked")
        entries.append("Medical necessity validated")
        entries.append("Fraud analysis performed")
        entries.append(
            f"Final decision rendered — {decision_data.get('decision_timestamp', now_str)}"
        )
        return entries
