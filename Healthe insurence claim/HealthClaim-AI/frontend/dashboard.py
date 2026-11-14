"""
HealthClaim AI — Streamlit Dashboard
Complete multi-page frontend for claims management and analytics.
"""

import os

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import io

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(
    page_title="HealthClaim AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.25rem;
}
.metric-card h2 { font-size: 2rem; margin: 0; }
.metric-card p  { font-size: 0.9rem; margin: 0; opacity: 0.85; }
.approved-badge   { background:#27ae60; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
.rejected-badge   { background:#e74c3c; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
.pending-badge    { background:#f39c12; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
.fraud-badge      { background:#8e44ad; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
.review-badge     { background:#2980b9; color:white; padding:4px 12px; border-radius:20px; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

for key, default in {
    "selected_claim_id": None,
    "claims_data": [],
    "fraud_stats": {},
    "claims_summary": {},
    "last_refresh": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_get(path: str, params: dict = None):
    try:
        r = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend. Is the FastAPI server running on port 8000?"
    except Exception as exc:
        return None, str(exc)


def api_post(path: str, json_data: dict = None, files=None, data=None):
    try:
        r = requests.post(
            f"{API_BASE_URL}{path}",
            json=json_data,
            files=files,
            data=data,
            timeout=30,
        )
        r.raise_for_status()
        return r.json(), None
    except requests.exceptions.ConnectionError:
        return None, "Cannot connect to backend."
    except Exception as exc:
        return None, str(exc)


def load_claims(status_filter=None):
    params = {"page": 1, "page_size": 200}
    if status_filter and status_filter != "All":
        params["status"] = status_filter
    data, err = api_get("/api/v1/claims/", params)
    if data:
        st.session_state.claims_data = data.get("claims", [])
    return err


def load_fraud_stats():
    data, err = api_get("/api/v1/analytics/fraud-stats")
    if data:
        st.session_state.fraud_stats = data
    return err


def load_claims_summary():
    data, err = api_get("/api/v1/analytics/claims-summary")
    if data:
        st.session_state.claims_summary = data
    return err


def status_badge(status_str: str) -> str:
    css_class = {
        "APPROVED": "approved-badge",
        "REJECTED": "rejected-badge",
        "PENDING": "pending-badge",
        "FRAUD_SUSPECTED": "fraud-badge",
        "UNDER_REVIEW": "review-badge",
    }.get(status_str, "pending-badge")
    return f'<span class="{css_class}">{status_str}</span>'


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.image("https://img.icons8.com/fluency/96/caduceus.png", width=60)
st.sidebar.title("HealthClaim AI")
st.sidebar.markdown("*Agentic Insurance Claims Processing*")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigate",
    ["🏠 Submit Claim", "📊 Claims Dashboard", "🔍 Claim Details", "📈 Analytics"],
    label_visibility="collapsed",
)

st.sidebar.divider()
if st.sidebar.button("🔄 Refresh Data"):
    err = load_claims()
    err2 = load_fraud_stats()
    err3 = load_claims_summary()
    if not (err or err2 or err3):
        st.sidebar.success("Data refreshed!")
    else:
        st.sidebar.warning("Refresh partial — check backend connection.")

# Check health
health, _ = api_get("/api/v1/health")
if health:
    st.sidebar.success("🟢 Backend Online")
else:
    st.sidebar.error("🔴 Backend Offline")


# ---------------------------------------------------------------------------
# Page 1 — Submit Claim
# ---------------------------------------------------------------------------

if page == "🏠 Submit Claim":
    st.title("🏥 Submit New Insurance Claim")
    st.markdown("Fill in all required fields and upload supporting documents.")

    with st.form("claim_form", clear_on_submit=False):
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Patient Information")
            patient_name = st.text_input("Patient Name *", placeholder="Jane Doe")
            patient_id = st.text_input("Patient ID *", placeholder="PAT-123456")
            patient_dob = st.date_input("Date of Birth *", value=datetime(1980, 1, 1))

        with col2:
            st.subheader("Provider Information")
            provider_name = st.text_input("Provider/Hospital Name *", placeholder="City Medical Center")
            provider_id = st.text_input("Provider ID *", placeholder="PRV-789")
            claim_date = st.date_input("Claim Date *", value=datetime.today())

        st.subheader("Clinical Details")
        col3, col4 = st.columns(2)
        with col3:
            diagnosis_codes = st.text_input(
                "ICD-10 Diagnosis Codes *",
                placeholder="J18.9, E11.9",
                help="Comma-separated ICD-10 codes",
            )
        with col4:
            procedure_codes = st.text_input(
                "CPT Procedure Codes *",
                placeholder="99213, 71046",
                help="Comma-separated CPT codes",
            )

        amount_claimed = st.number_input(
            "Amount Claimed (USD) *",
            min_value=0.0,
            step=100.0,
            format="%.2f",
        )
        clinical_notes = st.text_area(
            "Clinical Notes / Additional Information",
            height=120,
            placeholder="Describe the medical necessity, treatment details, etc.",
        )

        st.subheader("Supporting Documents")
        uploaded_files = st.file_uploader(
            "Upload medical records, bills, etc.",
            accept_multiple_files=True,
            type=["pdf", "png", "jpg", "jpeg", "txt"],
        )

        submitted = st.form_submit_button("🚀 Submit Claim", use_container_width=True, type="primary")

    if submitted:
        # Validate required fields
        errors = []
        if not patient_name.strip():
            errors.append("Patient Name is required.")
        if not patient_id.strip():
            errors.append("Patient ID is required.")
        if not provider_name.strip():
            errors.append("Provider Name is required.")
        if not provider_id.strip():
            errors.append("Provider ID is required.")
        if not diagnosis_codes.strip():
            errors.append("At least one diagnosis code is required.")
        if not procedure_codes.strip():
            errors.append("At least one procedure code is required.")
        if amount_claimed <= 0:
            errors.append("Amount claimed must be greater than 0.")

        if errors:
            for e in errors:
                st.error(e)
        else:
            with st.spinner("Submitting claim to HealthClaim AI…"):
                payload = {
                    "patient_name": patient_name.strip(),
                    "patient_id": patient_id.strip(),
                    "patient_dob": str(patient_dob),
                    "provider_name": provider_name.strip(),
                    "provider_id": provider_id.strip(),
                    "claim_date": str(claim_date),
                    "diagnosis_codes": [c.strip() for c in diagnosis_codes.split(",") if c.strip()],
                    "procedure_codes": [c.strip() for c in procedure_codes.split(",") if c.strip()],
                    "amount_claimed": amount_claimed,
                    "clinical_notes": clinical_notes.strip(),
                }

                result, err = api_post("/api/v1/claims/submit", json_data=payload)

                if err:
                    st.error(f"Submission failed: {err}")
                else:
                    claim_id = result["claim_id"]
                    st.success(f"✅ Claim submitted! **Claim ID:** `{claim_id}`")
                    st.info(f"**Claim Number:** {result['claim_number']} — Status: {result['status']}")
                    st.session_state.selected_claim_id = claim_id

                    # Upload documents
                    if uploaded_files:
                        progress = st.progress(0)
                        for i, f in enumerate(uploaded_files):
                            files = {"file": (f.name, f.getvalue(), f.type)}
                            _, ferr = api_post(
                                f"/api/v1/claims/{claim_id}/documents",
                                files=files,
                            )
                            if ferr:
                                st.warning(f"Document '{f.name}' upload failed: {ferr}")
                            progress.progress((i + 1) / len(uploaded_files))
                        st.success(f"{len(uploaded_files)} document(s) uploaded successfully.")

                    st.markdown("---")
                    st.markdown(
                        "Go to **🔍 Claim Details** in the sidebar to track processing status."
                    )


# ---------------------------------------------------------------------------
# Page 2 — Claims Dashboard
# ---------------------------------------------------------------------------

elif page == "📊 Claims Dashboard":
    st.title("📊 Claims Dashboard")

    # Load data
    with st.spinner("Loading claims…"):
        load_claims()
        load_fraud_stats()
        load_claims_summary()

    claims = st.session_state.claims_data
    summary = st.session_state.claims_summary
    fraud = st.session_state.fraud_stats

    # Metric cards
    by_status = summary.get("by_status", {})
    total = summary.get("total_claims", 0)
    approved = by_status.get("APPROVED", 0)
    rejected = by_status.get("REJECTED", 0)
    fraud_cnt = by_status.get("FRAUD_SUSPECTED", 0)
    pending = by_status.get("PENDING", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, value, color in [
        (c1, "Total Claims", total, "#3498db"),
        (c2, "✅ Approved", approved, "#27ae60"),
        (c3, "❌ Rejected", rejected, "#e74c3c"),
        (c4, "🚨 Fraud Suspected", fraud_cnt, "#8e44ad"),
        (c5, "⏳ Pending", pending, "#f39c12"),
    ]:
        col.markdown(
            f"""<div class="metric-card" style="background:{color}">
            <h2>{value}</h2><p>{label}</p></div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if not claims:
        st.info("No claims found. Submit your first claim to get started.")
        st.stop()

    df = pd.DataFrame(claims)

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Claims by Status")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        fig_bar = px.bar(
            status_counts,
            x="Status",
            y="Count",
            color="Status",
            color_discrete_map={
                "APPROVED": "#27ae60",
                "REJECTED": "#e74c3c",
                "PENDING": "#f39c12",
                "FRAUD_SUSPECTED": "#8e44ad",
                "UNDER_REVIEW": "#2980b9",
            },
        )
        fig_bar.update_layout(showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        # 30-day trend
        st.subheader("Claims Trend (30 days)")
        daily = summary.get("daily_trend_last_30_days", {})
        if daily:
            trend_df = pd.DataFrame(list(daily.items()), columns=["Date", "Count"])
            trend_df["Date"] = pd.to_datetime(trend_df["Date"])
            trend_df = trend_df.sort_values("Date")
            fig_line = px.line(trend_df, x="Date", y="Count", markers=True)
            fig_line.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_line, use_container_width=True)
        else:
            st.info("No trend data available yet.")

    with col_c:
        st.subheader("Fraud Risk Distribution")
        if "fraud_score" in df.columns and df["fraud_score"].notna().any():
            bins = [0, 0.3, 0.5, 0.7, 1.0]
            labels = ["Low (<0.3)", "Moderate (0.3–0.5)", "High (0.5–0.7)", "Critical (>0.7)"]
            df["risk_band"] = pd.cut(
                df["fraud_score"].fillna(0), bins=bins, labels=labels, right=True
            )
            risk_counts = df["risk_band"].value_counts().reset_index()
            risk_counts.columns = ["Band", "Count"]
            fig_pie = px.pie(
                risk_counts,
                names="Band",
                values="Count",
                color_discrete_sequence=["#27ae60", "#f39c12", "#e67e22", "#e74c3c"],
            )
            fig_pie.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Fraud scores not available yet.")

    # Data table
    st.subheader("All Claims")
    status_options = ["All"] + list(df["status"].unique())
    sel_status = st.selectbox("Filter by Status", status_options)
    filtered_df = df if sel_status == "All" else df[df["status"] == sel_status]

    display_cols = [c for c in
                    ["claim_number", "patient_name", "provider_name", "claim_date",
                     "amount_claimed", "amount_approved", "status", "fraud_score"]
                    if c in filtered_df.columns]

    st.dataframe(
        filtered_df[display_cols].rename(columns={
            "claim_number": "Claim #",
            "patient_name": "Patient",
            "provider_name": "Provider",
            "claim_date": "Date",
            "amount_claimed": "Claimed ($)",
            "amount_approved": "Approved ($)",
            "status": "Status",
            "fraud_score": "Fraud Score",
        }),
        use_container_width=True,
        height=400,
    )


# ---------------------------------------------------------------------------
# Page 3 — Claim Details
# ---------------------------------------------------------------------------

elif page == "🔍 Claim Details":
    st.title("🔍 Claim Details")

    claims, err = api_get("/api/v1/claims/", {"page": 1, "page_size": 500})
    claim_list = claims.get("claims", []) if claims else []

    claim_numbers = {c["claim_number"]: c["claim_id"] for c in claim_list}

    col_input, col_select = st.columns([1, 2])
    with col_input:
        manual_id = st.text_input("Enter Claim ID", placeholder="UUID…")
    with col_select:
        selected_num = st.selectbox(
            "Or select Claim Number",
            options=[""] + list(claim_numbers.keys()),
        )

    claim_id = manual_id.strip() or claim_numbers.get(selected_num, "") or st.session_state.selected_claim_id

    if not claim_id:
        st.info("Enter a Claim ID or select a claim above.")
        st.stop()

    with st.spinner("Loading claim details…"):
        claim_data, err = api_get(f"/api/v1/claims/{claim_id}")

    if err:
        st.error(f"Error: {err}")
        st.stop()
    if not claim_data:
        st.warning("Claim not found.")
        st.stop()

    # Header status card
    s = claim_data.get("status", "PENDING")
    color_map = {
        "APPROVED": "#27ae60", "REJECTED": "#e74c3c",
        "PENDING": "#f39c12", "FRAUD_SUSPECTED": "#8e44ad", "UNDER_REVIEW": "#2980b9",
    }
    bg = color_map.get(s, "#7f8c8d")
    st.markdown(
        f"""<div style="background:{bg};color:white;padding:1.2rem;border-radius:10px;text-align:center;margin-bottom:1rem">
        <h2 style="margin:0">{s}</h2>
        <p style="margin:0;opacity:0.85">{claim_data.get('claim_number','')}</p></div>""",
        unsafe_allow_html=True,
    )

    # Sections
    with st.expander("👤 Patient & Provider Info", expanded=True):
        p = claim_data.get("patient", {})
        pr = claim_data.get("provider", {})
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Patient Name:** {p.get('name')}")
            st.write(f"**Patient ID:** {p.get('id')}")
            st.write(f"**Date of Birth:** {p.get('dob')}")
        with c2:
            st.write(f"**Provider Name:** {pr.get('name')}")
            st.write(f"**Provider ID:** {pr.get('id')}")
            st.write(f"**Claim Date:** {claim_data.get('claim_date')}")

    with st.expander("📋 Policy Coverage Analysis"):
        fin = claim_data.get("financials", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("Amount Claimed", f"${fin.get('amount_claimed', 0):,.2f}")
        c2.metric("Amount Approved", f"${fin.get('amount_approved', 0):,.2f}")
        savings = (fin.get('amount_claimed', 0) or 0) - (fin.get('amount_approved', 0) or 0)
        c3.metric("Reduction", f"${savings:,.2f}")
        st.write(f"**Diagnosis Codes:** {', '.join(claim_data.get('diagnosis_codes', []))}")
        st.write(f"**Procedure Codes:** {', '.join(claim_data.get('procedure_codes', []))}")

    with st.expander("🩺 Medical Validation"):
        st.write(f"**AI Decision:** {claim_data.get('ai_decision', 'N/A')}")
        st.write(f"**AI Reasoning:** {claim_data.get('ai_reasoning', 'N/A')}")

    with st.expander("🚨 Fraud Assessment"):
        fraud = claim_data.get("fraud_analysis", {})
        score = fraud.get("fraud_score") or 0.0
        flags = fraud.get("fraud_flags", [])

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            title={"text": "Fraud Risk Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.3], "color": "#27ae60"},
                    {"range": [0.3, 0.5], "color": "#f39c12"},
                    {"range": [0.5, 0.7], "color": "#e67e22"},
                    {"range": [0.7, 1.0], "color": "#e74c3c"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 0.7},
            },
        ))
        fig_gauge.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        if flags:
            st.warning("**Fraud Flags Detected:**")
            for flag in flags:
                st.write(f"• {flag}")
        else:
            st.success("No fraud flags detected.")

    with st.expander("📄 Documents"):
        docs = claim_data.get("documents", [])
        if docs:
            for doc in docs:
                st.write(
                    f"📎 **{doc['filename']}** — {doc['file_type']} "
                    f"({doc.get('size_bytes', 0):,} bytes) "
                    f"{'✅ Verified' if doc['is_verified'] else '⏳ Pending'}"
                )
        else:
            st.info("No documents uploaded.")

    with st.expander("📜 Audit Trail"):
        audit = claim_data.get("audit_trail", [])
        if audit:
            for entry in audit:
                st.write(f"**{entry['timestamp']}** — `{entry['actor']}` → {entry['action']}")
        else:
            st.info("No audit entries.")

    # PDF Download
    st.markdown("---")
    if st.button("📥 Download PDF Report", type="primary"):
        try:
            resp = requests.get(f"{API_BASE_URL}/api/v1/claims/{claim_id}/report", timeout=15)
            if resp.status_code == 200:
                st.download_button(
                    label="💾 Save PDF",
                    data=resp.content,
                    file_name=f"claim_report_{claim_id[:8]}.pdf",
                    mime="application/pdf",
                )
            else:
                st.error(f"Report unavailable (HTTP {resp.status_code}). Try again shortly.")
        except Exception as exc:
            st.error(f"Download failed: {exc}")

    # Manual review
    st.markdown("---")
    st.subheader("🔧 Manual Review Override")
    with st.form("review_form"):
        new_status = st.selectbox(
            "New Status",
            ["APPROVED", "REJECTED", "UNDER_REVIEW", "FRAUD_SUSPECTED", "PENDING"],
        )
        reason = st.text_area("Reason / Notes", height=80)
        adjuster_id = st.text_input("Adjuster ID", value="adjuster-001")
        submit_review = st.form_submit_button("Apply Override")

    if submit_review:
        if not reason.strip():
            st.warning("Please provide a reason for the override.")
        else:
            with st.spinner("Applying override…"):
                try:
                    r = requests.post(
                        f"{API_BASE_URL}/api/v1/claims/{claim_id}/review",
                        params={"new_status": new_status, "reason": reason, "adjuster_id": adjuster_id},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success(f"Status updated to **{new_status}**.")
                        st.rerun()
                    else:
                        st.error(f"Override failed: {r.text}")
                except Exception as exc:
                    st.error(f"Error: {exc}")


# ---------------------------------------------------------------------------
# Page 4 — Analytics
# ---------------------------------------------------------------------------

elif page == "📈 Analytics":
    st.title("📈 Analytics & Insights")

    with st.spinner("Loading analytics…"):
        load_fraud_stats()
        load_claims_summary()
        load_claims()

    fraud = st.session_state.fraud_stats
    summary = st.session_state.claims_summary
    claims = st.session_state.claims_data

    # Top KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Claims", fraud.get("total_claims", 0))
    c2.metric("Fraud Rate", f"{fraud.get('fraud_rate', 0)*100:.1f}%")
    c3.metric("Avg Fraud Score", f"{fraud.get('avg_fraud_score', 0):.3f}")
    c4.metric(
        "Approval Rate",
        f"{summary.get('approval_rate', 0)*100:.1f}%",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("🚩 Top Fraud Flags")
        top_flags = fraud.get("top_fraud_flags", [])
        if top_flags:
            flags_df = pd.DataFrame(top_flags)
            fig_flags = px.bar(
                flags_df,
                x="count",
                y="flag",
                orientation="h",
                color="count",
                color_continuous_scale="Reds",
            )
            fig_flags.update_layout(margin=dict(t=10, b=10), showlegend=False)
            st.plotly_chart(fig_flags, use_container_width=True)
        else:
            st.info("No fraud flags recorded yet.")

    with col_right:
        st.subheader("💰 Amount Analysis")
        total_claimed = summary.get("total_amount_claimed", 0)
        total_approved = summary.get("total_amount_approved", 0)
        total_rejected = total_claimed - total_approved

        fig_amounts = go.Figure(data=[
            go.Bar(name="Total Claimed", x=["Amounts"], y=[total_claimed], marker_color="#3498db"),
            go.Bar(name="Total Approved", x=["Amounts"], y=[total_approved], marker_color="#27ae60"),
            go.Bar(name="Reduction", x=["Amounts"], y=[total_rejected], marker_color="#e74c3c"),
        ])
        fig_amounts.update_layout(barmode="group", margin=dict(t=10, b=10))
        st.plotly_chart(fig_amounts, use_container_width=True)

    # Provider risk ranking
    st.subheader("🏥 Provider Risk Ranking")
    if claims:
        df = pd.DataFrame(claims)
        if "fraud_score" in df.columns and "provider_name" in df.columns:
            provider_risk = (
                df.groupby("provider_name")["fraud_score"]
                .agg(["mean", "count"])
                .reset_index()
                .rename(columns={"mean": "avg_fraud_score", "count": "claim_count"})
                .sort_values("avg_fraud_score", ascending=False)
                .head(10)
            )
            fig_prov = px.bar(
                provider_risk,
                x="avg_fraud_score",
                y="provider_name",
                orientation="h",
                color="avg_fraud_score",
                color_continuous_scale="RdYlGn_r",
                hover_data=["claim_count"],
            )
            fig_prov.update_layout(margin=dict(t=10, b=10))
            st.plotly_chart(fig_prov, use_container_width=True)
        else:
            st.info("Insufficient data for provider risk analysis.")

    # Monthly volume
    st.subheader("📅 Monthly Claims Volume")
    daily = summary.get("daily_trend_last_30_days", {})
    if daily:
        trend_df = pd.DataFrame(list(daily.items()), columns=["Date", "Count"])
        trend_df["Date"] = pd.to_datetime(trend_df["Date"])
        trend_df["Month"] = trend_df["Date"].dt.to_period("M").astype(str)
        monthly = trend_df.groupby("Month")["Count"].sum().reset_index()
        fig_monthly = px.bar(monthly, x="Month", y="Count", color="Count",
                             color_continuous_scale="Blues")
        fig_monthly.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_monthly, use_container_width=True)
    else:
        st.info("No trend data available.")

    # Processing time (simulated)
    st.subheader("⏱️ Processing Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Avg Processing Time", "~2.3 min")
    m2.metric("Auto-Approved Rate", f"{summary.get('approval_rate', 0)*100:.1f}%")
    m3.metric("Fraud Detected", fraud.get("fraud_detected_count", 0))
