"""
HealthClaim AI - Main FastAPI Application
Agentic AI-Based Healthcare Insurance Claims & Fraud Detection System
"""

import os
import uuid
import shutil
import json
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, case

from app.database.db import get_db, create_all_tables, engine
from app.database import models as db_models
from app.schemas.claim_schema import ClaimCreate
from app.utils.logger import get_logger
from app.utils.config import settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup → yield → shutdown."""
    logger.info("HealthClaim AI starting up…")

    # 1. Create all DB tables
    try:
        create_all_tables()
        logger.info("Database tables created/verified.")
    except Exception as exc:
        logger.error(f"DB init failed: {exc}")

    # 2. Ensure data directories exist
    Path("data/claims").mkdir(parents=True, exist_ok=True)
    Path("data/policies").mkdir(parents=True, exist_ok=True)
    Path("data/medical_documents").mkdir(parents=True, exist_ok=True)
    Path("app/models").mkdir(parents=True, exist_ok=True)

    # 3. Initialise ChromaDB (best-effort)
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        client.get_or_create_collection("policy_docs")
        client.get_or_create_collection("medical_codes")
        logger.info("ChromaDB initialised.")
    except Exception as exc:
        logger.warning(f"ChromaDB init skipped: {exc}")

    # 4. Load fraud ML model (best-effort)
    try:
        import joblib
        model_path = Path("app/models/fraud_model.pkl")
        if model_path.exists():
            app.state.fraud_model = joblib.load(model_path)
            logger.info("Fraud ML model loaded.")
        else:
            app.state.fraud_model = None
            logger.warning("fraud_model.pkl not found; running without ML model.")
    except Exception as exc:
        app.state.fraud_model = None
        logger.warning(f"Fraud model load failed: {exc}")

    # 5. Seed sample policies (best-effort)
    try:
        policy_file = Path("data/policies/sample_policy.json")
        if policy_file.exists():
            with open(policy_file) as f:
                policies = json.load(f)
            db = next(get_db())
            for pol in policies:
                exists = db.query(db_models.Policy).filter_by(
                    policy_number=pol["policy_number"]
                ).first()
                if not exists:
                    db.add(db_models.Policy(**pol))
            db.commit()
            db.close()
            logger.info("Sample policies seeded.")
    except Exception as exc:
        logger.warning(f"Policy seeding skipped: {exc}")

    yield

    logger.info("HealthClaim AI shutting down.")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HealthClaim AI",
    description="Agentic AI-Based Healthcare Insurance Claims & Fraud Detection System",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------

def _run_claim_workflow(claim_id: str, db_session_factory):
    """Background task: run full agentic claim processing workflow."""
    db = db_session_factory()
    try:
        claim = db.query(db_models.Claim).filter_by(id=claim_id).first()
        if not claim:
            logger.error(f"Claim {claim_id} not found for background processing.")
            return

        # Attempt to invoke the workflow; fall back to a heuristic decision
        try:
            from app.workflows.claim_workflow import ClaimProcessingWorkflow
            workflow = ClaimProcessingWorkflow()
            result = workflow.run(claim_id, db)
        except Exception as exc:
            logger.warning(f"Workflow invocation failed ({exc}); applying heuristic.")
            # Simple heuristic fallback
            fraud_score = 0.3
            if claim.amount_claimed and claim.amount_claimed > 50_000:
                fraud_score = 0.75
            result = {
                "status": "FRAUD_SUSPECTED" if fraud_score > settings.fraud_threshold else "APPROVED",
                "fraud_score": fraud_score,
                "fraud_flags": ["High amount"] if fraud_score > settings.fraud_threshold else [],
                "ai_decision": "AUTO",
                "ai_reasoning": "Heuristic fallback decision (workflow unavailable).",
                "amount_approved": claim.amount_claimed * (1 - fraud_score),
            }

        # Persist result
        claim.status = result.get("status", "APPROVED")
        claim.fraud_score = result.get("fraud_score", 0.0)
        claim.fraud_flags = json.dumps(result.get("fraud_flags", []))
        claim.ai_decision = result.get("ai_decision", "")
        claim.ai_reasoning = result.get("ai_reasoning", "")
        claim.amount_approved = result.get("amount_approved", claim.amount_claimed)
        claim.updated_at = datetime.utcnow()
        db.commit()
        logger.info(f"Claim {claim_id} processed → {claim.status}")

        # Audit log
        db.add(db_models.AuditLog(
            id=str(uuid.uuid4()),
            claim_id=claim_id,
            action="PROCESSED",
            actor=None,
            details=json.dumps(result),
        ))
        db.commit()

    except Exception as exc:
        logger.error(f"Background processing error for {claim_id}: {exc}")
        db.rollback()
    finally:
        db.close()


def _extract_document(doc_id: str, file_path: str, claim_id: str, db_session_factory):
    """Background task: OCR / text extraction for uploaded document."""
    db = db_session_factory()
    try:
        text = ""
        path = Path(file_path)
        if path.suffix.lower() == ".pdf":
            try:
                import pdfplumber
                with pdfplumber.open(path) as pdf:
                    text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            except Exception:
                pass
        elif path.suffix.lower() in (".png", ".jpg", ".jpeg", ".tiff"):
            try:
                import pytesseract
                from PIL import Image
                text = pytesseract.image_to_string(Image.open(path))
            except Exception:
                pass
        else:
            try:
                text = path.read_text(errors="replace")
            except Exception:
                pass

        doc = db.query(db_models.Document).filter_by(id=doc_id).first()
        if doc:
            doc.extracted_text = text
            doc.is_verified = True
            db.commit()
            logger.info(f"Document {doc_id} extraction complete ({len(text)} chars).")
    except Exception as exc:
        logger.error(f"Document extraction error {doc_id}: {exc}")
    finally:
        db.close()


def _generate_pdf_report(claim_id: str, db_session_factory):
    """Generate a PDF report for the claim."""
    db = db_session_factory()
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib import colors

        claim = db.query(db_models.Claim).filter_by(id=claim_id).first()
        if not claim:
            return

        report_dir = Path(f"data/claims/{claim_id}")
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "report.pdf"

        c = rl_canvas.Canvas(str(report_path), pagesize=letter)
        width, height = letter

        # Header
        c.setFillColor(colors.HexColor("#1a237e"))
        c.rect(0, height - 80, width, 80, fill=True, stroke=False)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(40, height - 50, "HealthClaim AI — Insurance Claim Report")
        c.setFont("Helvetica", 11)
        c.drawString(40, height - 68, f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")

        # Content
        c.setFillColor(colors.black)
        y = height - 110

        def row(label, value, indent=40):
            nonlocal y
            if y < 80:
                c.showPage()
                y = height - 60
            c.setFont("Helvetica-Bold", 10)
            c.drawString(indent, y, f"{label}:")
            c.setFont("Helvetica", 10)
            c.drawString(indent + 160, y, str(value) if value is not None else "N/A")
            y -= 18

        row("Claim ID", claim.id)
        row("Claim Number", claim.claim_number)
        row("Status", claim.status)
        row("Patient Name", claim.patient_name)
        row("Patient ID", claim.patient_id)
        row("Provider Name", claim.provider_name)
        row("Claim Date", str(claim.claim_date))
        row("Amount Claimed", f"${claim.amount_claimed:,.2f}" if claim.amount_claimed else "N/A")
        row("Amount Approved", f"${claim.amount_approved:,.2f}" if claim.amount_approved else "N/A")
        row("Fraud Score", f"{claim.fraud_score:.2f}" if claim.fraud_score is not None else "N/A")
        row("AI Decision", claim.ai_decision)
        y -= 10
        c.setFont("Helvetica-Bold", 10)
        c.drawString(40, y, "AI Reasoning:")
        y -= 16
        c.setFont("Helvetica", 9)
        reasoning = claim.ai_reasoning or "N/A"
        for chunk in [reasoning[i:i+90] for i in range(0, min(len(reasoning), 540), 90)]:
            c.drawString(50, y, chunk)
            y -= 14

        c.save()
        logger.info(f"PDF report generated at {report_path}")
    except Exception as exc:
        logger.error(f"PDF generation failed for {claim_id}: {exc}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Session factory used by background tasks
# ---------------------------------------------------------------------------

from sqlalchemy.orm import sessionmaker
BackgroundSessionFactory = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", tags=["Health"])
def health_check():
    """Service health check."""
    return {
        "status": "healthy",
        "service": "HealthClaim AI",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "fraud_model_loaded": app.state.fraud_model is not None,
    }


# --- Claims ---

@app.post("/api/v1/claims/submit", status_code=status.HTTP_202_ACCEPTED, tags=["Claims"])
def submit_claim(
    claim_data: ClaimCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Submit a new insurance claim for AI processing."""
    claim_id = str(uuid.uuid4())
    claim_number = f"CLM-{datetime.utcnow().strftime('%Y%m%d')}-{claim_id[:8].upper()}"

    new_claim = db_models.Claim(
        id=claim_id,
        claim_number=claim_number,
        patient_name=claim_data.patient_name,
        patient_id=claim_data.patient_id,
        patient_dob=claim_data.patient_dob,
        provider_name=claim_data.provider_name,
        provider_id=claim_data.provider_id,
        claim_date=claim_data.claim_date or datetime.utcnow(),
        diagnosis_codes=json.dumps(claim_data.diagnosis_codes),
        procedure_codes=json.dumps(claim_data.procedure_codes),
        amount_claimed=claim_data.amount_claimed,
        status="PENDING",
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    if claim_data.clinical_notes:
        new_claim.ai_reasoning = claim_data.clinical_notes

    db.add(new_claim)
    db.commit()
    db.refresh(new_claim)

    # Audit
    db.add(db_models.AuditLog(
        id=str(uuid.uuid4()),
        claim_id=claim_id,
        action="SUBMITTED",
        actor=None,
        details=json.dumps({"claim_number": claim_number}),
    ))
    db.commit()

    # Queue background processing
    background_tasks.add_task(_run_claim_workflow, claim_id, BackgroundSessionFactory)
    background_tasks.add_task(_generate_pdf_report, claim_id, BackgroundSessionFactory)

    return {
        "claim_id": claim_id,
        "claim_number": claim_number,
        "status": "PENDING",
        "message": "Claim submitted successfully. AI processing has started.",
    }


@app.post("/api/v1/claims/{claim_id}/documents", tags=["Claims"])
async def upload_document(
    claim_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload a supporting document for a claim."""
    claim = db.query(db_models.Claim).filter_by(id=claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

    # Save file
    claim_dir = Path(f"data/claims/{claim_id}")
    claim_dir.mkdir(parents=True, exist_ok=True)

    doc_id = str(uuid.uuid4())
    safe_filename = f"{doc_id}_{file.filename}"
    file_path = claim_dir / safe_filename

    content = await file.read()
    file_path.write_bytes(content)

    # DB record
    doc = db_models.Document(
        id=doc_id,
        claim_id=claim_id,
        filename=file.filename,
        file_type=file.content_type or "application/octet-stream",
        file_path=str(file_path),
        file_size=len(content),
        is_verified=False,
        created_at=datetime.utcnow(),
    )
    db.add(doc)
    db.commit()

    # Async extraction
    background_tasks.add_task(
        _extract_document, doc_id, str(file_path), claim_id, BackgroundSessionFactory
    )

    return {
        "document_id": doc_id,
        "filename": file.filename,
        "size_bytes": len(content),
        "claim_id": claim_id,
        "status": "uploaded",
        "message": "Document uploaded. Text extraction is in progress.",
    }


@app.get("/api/v1/claims/{claim_id}", tags=["Claims"])
def get_claim(claim_id: str, db: Session = Depends(get_db)):
    """Get full claim details including all processing results."""
    claim = db.query(db_models.Claim).filter_by(id=claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

    documents = db.query(db_models.Document).filter_by(claim_id=claim_id).all()
    audit_logs = db.query(db_models.AuditLog).filter_by(claim_id=claim_id).order_by(
        db_models.AuditLog.timestamp
    ).all()

    return {
        "claim_id": claim.id,
        "claim_number": claim.claim_number,
        "status": claim.status,
        "patient": {
            "name": claim.patient_name,
            "id": claim.patient_id,
            "dob": str(claim.patient_dob) if claim.patient_dob else None,
        },
        "provider": {
            "name": claim.provider_name,
            "id": claim.provider_id,
        },
        "claim_date": str(claim.claim_date),
        "diagnosis_codes": json.loads(claim.diagnosis_codes) if claim.diagnosis_codes else [],
        "procedure_codes": json.loads(claim.procedure_codes) if claim.procedure_codes else [],
        "financials": {
            "amount_claimed": claim.amount_claimed,
            "amount_approved": claim.amount_approved,
        },
        "fraud_analysis": {
            "fraud_score": claim.fraud_score,
            "fraud_flags": json.loads(claim.fraud_flags) if claim.fraud_flags else [],
        },
        "ai_decision": claim.ai_decision,
        "ai_reasoning": claim.ai_reasoning,
        "documents": [
            {
                "document_id": d.id,
                "filename": d.filename,
                "file_type": d.file_type,
                "size_bytes": d.file_size,
                "is_verified": d.is_verified,
            }
            for d in documents
        ],
        "audit_trail": [
            {
                "action": a.action,
                "actor": a.actor,
                "timestamp": str(a.timestamp),
                "details": json.loads(a.details) if a.details else {},
            }
            for a in audit_logs
        ],
        "created_at": str(claim.created_at),
        "updated_at": str(claim.updated_at),
    }


@app.get("/api/v1/claims/{claim_id}/report", tags=["Claims"])
def get_claim_report(
    claim_id: str,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """Download the generated PDF report for a claim."""
    claim = db.query(db_models.Claim).filter_by(id=claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

    report_path = Path(f"data/claims/{claim_id}/report.pdf")
    if not report_path.exists():
        # Regenerate on the fly (synchronous, small doc)
        _generate_pdf_report(claim_id, BackgroundSessionFactory)
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not yet available. Try again shortly.")

    return FileResponse(
        path=str(report_path),
        media_type="application/pdf",
        filename=f"claim_report_{claim.claim_number}.pdf",
    )


@app.get("/api/v1/claims/", tags=["Claims"])
def list_claims(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, alias="status"),
    db: Session = Depends(get_db),
):
    """List all claims with pagination and optional status filter."""
    query = db.query(db_models.Claim)
    if status_filter:
        query = query.filter(db_models.Claim.status == status_filter.upper())

    total = query.count()
    claims = (
        query.order_by(db_models.Claim.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size,
        "claims": [
            {
                "claim_id": c.id,
                "claim_number": c.claim_number,
                "patient_name": c.patient_name,
                "provider_name": c.provider_name,
                "claim_date": str(c.claim_date),
                "amount_claimed": c.amount_claimed,
                "amount_approved": c.amount_approved,
                "status": c.status,
                "fraud_score": c.fraud_score,
                "created_at": str(c.created_at),
            }
            for c in claims
        ],
    }


@app.post("/api/v1/claims/{claim_id}/review", tags=["Claims"])
def review_claim(
    claim_id: str,
    new_status: str,
    reason: str,
    adjuster_id: str = "adjuster",
    db: Session = Depends(get_db),
):
    """Manual override for claim decision by an adjuster."""
    valid_statuses = {"APPROVED", "REJECTED", "UNDER_REVIEW", "FRAUD_SUSPECTED", "PENDING"}
    if new_status.upper() not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status '{new_status}'. Must be one of: {valid_statuses}",
        )

    claim = db.query(db_models.Claim).filter_by(id=claim_id).first()
    if not claim:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found.")

    old_status = claim.status
    claim.status = new_status.upper()
    claim.ai_decision = "MANUAL_OVERRIDE"
    claim.ai_reasoning = f"Manual review by {adjuster_id}: {reason}"
    claim.updated_at = datetime.utcnow()
    db.commit()

    db.add(db_models.AuditLog(
        id=str(uuid.uuid4()),
        claim_id=claim_id,
        action="MANUAL_REVIEW",
        actor=None,
        details=json.dumps({
            "actor": adjuster_id,
            "old_status": old_status,
            "new_status": new_status.upper(),
            "reason": reason,
        }),
    ))
    db.commit()

    return {
        "claim_id": claim_id,
        "old_status": old_status,
        "new_status": new_status.upper(),
        "adjuster_id": adjuster_id,
        "reason": reason,
        "updated_at": claim.updated_at.isoformat(),
    }


# --- Analytics ---

@app.get("/api/v1/analytics/fraud-stats", tags=["Analytics"])
def get_fraud_stats(db: Session = Depends(get_db)):
    """Fraud detection statistics across all claims."""
    total = db.query(db_models.Claim).count()
    fraud_detected = db.query(db_models.Claim).filter(
        db_models.Claim.status == "FRAUD_SUSPECTED"
    ).count()

    avg_fraud_score_row = db.query(func.avg(db_models.Claim.fraud_score)).scalar()
    avg_fraud_score = round(float(avg_fraud_score_row), 4) if avg_fraud_score_row else 0.0

    # Aggregate fraud flags
    all_flags: Dict[str, int] = {}
    for (flags_json,) in db.query(db_models.Claim.fraud_flags).filter(
        db_models.Claim.fraud_flags.isnot(None)
    ).all():
        try:
            for flag in json.loads(flags_json or "[]"):
                all_flags[flag] = all_flags.get(flag, 0) + 1
        except Exception:
            pass

    top_flags = sorted(all_flags.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "total_claims": total,
        "fraud_detected_count": fraud_detected,
        "fraud_rate": round(fraud_detected / total, 4) if total else 0.0,
        "avg_fraud_score": avg_fraud_score,
        "top_fraud_flags": [{"flag": f, "count": c} for f, c in top_flags],
    }


@app.get("/api/v1/analytics/claims-summary", tags=["Analytics"])
def get_claims_summary(
    start_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="YYYY-MM-DD"),
    db: Session = Depends(get_db),
):
    """Claims summary: by status, by date range, amounts."""
    query = db.query(db_models.Claim)

    if start_date:
        try:
            query = query.filter(
                db_models.Claim.created_at >= datetime.strptime(start_date, "%Y-%m-%d")
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid start_date format. Use YYYY-MM-DD.")
    if end_date:
        try:
            query = query.filter(
                db_models.Claim.created_at <= datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
            )
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid end_date format. Use YYYY-MM-DD.")

    claims = query.all()

    by_status: Dict[str, int] = {}
    total_claimed = 0.0
    total_approved = 0.0

    for c in claims:
        by_status[c.status] = by_status.get(c.status, 0) + 1
        total_claimed += c.amount_claimed or 0.0
        total_approved += c.amount_approved or 0.0

    # Daily trend (last 30 days)
    cutoff = datetime.utcnow() - timedelta(days=30)
    daily: Dict[str, int] = {}
    for c in claims:
        if c.created_at and c.created_at >= cutoff:
            day = c.created_at.strftime("%Y-%m-%d")
            daily[day] = daily.get(day, 0) + 1

    return {
        "total_claims": len(claims),
        "by_status": by_status,
        "total_amount_claimed": round(total_claimed, 2),
        "total_amount_approved": round(total_approved, 2),
        "approval_rate": round(
            by_status.get("APPROVED", 0) / len(claims), 4
        ) if claims else 0.0,
        "daily_trend_last_30_days": daily,
        "date_range": {"start": start_date, "end": end_date},
    }
