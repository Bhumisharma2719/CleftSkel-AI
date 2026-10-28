"""FastAPI API for CLEFTSKEL-AI research predictions and reports.

Growth timing text is a transparent stage-based research heuristic. It is not a
patient-specific treatment recommendation or a substitute for clinician review.
"""

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image as PILImage, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool

from model import predict_from_image, warmup_model

logger = logging.getLogger("cleftskel")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Pay model loading / first-kernel setup during server startup when artifacts exist.
    try:
        warmed = await run_in_threadpool(warmup_model)
        if warmed:
            logger.info("CLEFTSKEL-AI model loaded and warmed up.")
    except Exception:
        # Keep health/docs available if the model artifact or accelerator is not ready.
        logger.exception("CLEFTSKEL-AI model warm-up failed; inference will retry on demand.")
    yield

app = FastAPI(
    title="CLEFTSKEL-AI API",
    description="Research prototype for lateral cephalogram CVM classification.",
    version="0.4.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 15 * 1024 * 1024
HOSPITAL_NAME = os.getenv("CLEFTSKEL_HOSPITAL_NAME", "CLEFTSKEL-AI Research Center")
GUIDANCE_HIGH_CONFIDENCE = float(os.getenv("CLEFTSKEL_GUIDANCE_HIGH_CONFIDENCE", "0.65"))
GUIDANCE_SECONDARY_REVIEW_CONFIDENCE = float(os.getenv("CLEFTSKEL_GUIDANCE_SECONDARY_REVIEW", "0.40"))
if not 0 <= GUIDANCE_SECONDARY_REVIEW_CONFIDENCE < GUIDANCE_HIGH_CONFIDENCE <= 1:
    raise ValueError("Guidance thresholds must satisfy 0 <= secondary-review < high-confidence <= 1.")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CLEFTSKEL-AI API"}


async def _read_image(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")
    image_bytes = await file.read(MAX_FILE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB limit.")
    return image_bytes


async def _predict_or_http_error(image_bytes: bytes) -> dict:
    try:
        result = await run_in_threadpool(predict_from_image, image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc
    result["clinical_guidance"] = build_growth_guidance(result)
    return result


def build_growth_guidance(result: dict) -> dict:
    """Map CVM stages to cautious, qualitative growth-phase review prompts.

    This intentionally does not calculate a numeric maturity countdown. A
    single CVM stage and a synthetic stage-mapped age target cannot establish
    an individual growth-completion date.
    """
    stage = result.get("cvm_stage")
    confidence = float(result.get("cvm_stage_confidence") or 0.0)
    age = result.get("predicted_skeletal_age_years")
    age_source = result.get("age_target_source")
    top2 = result.get("cvm_stage_top2") or []
    age_note = (
        "Synthetic stage-mapped proxy; excluded from the timing rule."
        if age is not None and age_source == "synthetic_stage_mapping"
        else ("Age estimate is shown as context only; no validated age-stage model is available to convert it into a growth-duration estimate." if age is not None else "No skeletal-age estimate is available.")
    )
    confidence_band = (
        "higher_raw_score" if confidence >= GUIDANCE_HIGH_CONFIDENCE
        else ("intermediate_review" if confidence >= GUIDANCE_SECONDARY_REVIEW_CONFIDENCE else "low_raw_score")
    )

    def phase_for(candidate_stage: int) -> tuple[str, str]:
        if candidate_stage in (1, 2):
            return "pre_peak", "Pre-pubertal deceleration / pre-peak phase"
        if candidate_stage in (3, 4):
            return "circumpubertal_peak", "Peak circumpubertal velocity window"
        return "postpubertal", "Post-pubertal deceleration"

    candidate_phases = []
    for candidate in top2:
        candidate_stage = candidate.get("stage")
        if candidate_stage in {1, 2, 3, 4, 5, 6}:
            phase_key, phase_label = phase_for(candidate_stage)
            candidate_phases.append({
                "stage": candidate_stage,
                "label": candidate.get("label", f"CVM-S{candidate_stage}"),
                "softmax_score_percent": candidate.get("softmax_score_percent"),
                "growth_phase": phase_key,
                "growth_velocity_descriptor": phase_label,
            })

    def score_breakdown_text() -> str:
        if not top2:
            return "Top-two stage scores were not supplied by the model."
        return "; ".join(
            f"{item.get('label', 'CVM stage')} {float(item.get('softmax_score_percent', 0)):.1f}%"
            for item in top2[:2]
        ) + " (raw, uncalibrated softmax scores)."

    base = {
        "rules_version": "cvm-growth-window-0.2",
        "confidence_band": confidence_band,
        "confidence_calibration": result.get("confidence_calibration", "uncalibrated_softmax_scores"),
        "cvm_stage_top2": top2[:2],
        "candidate_growth_phases": candidate_phases,
        "growth_velocity_estimate_mm_per_year": None,
        "growth_velocity_measurement_note": "Velocity in mm/year requires serial, longitudinal measurements; it is not calculated from a single image.",
        "active_growth_duration_estimate_years": None,
        "active_growth_duration_note": "No numeric duration is estimated: a single CVM image and current age head are not validated for patient-specific time-to-maturity prediction.",
        "age_context": age_note,
        "clinician_validation_required": True,
        "confidence_thresholds": {
            "secondary_review_minimum": GUIDANCE_SECONDARY_REVIEW_CONFIDENCE,
            "higher_raw_score_minimum": GUIDANCE_HIGH_CONFIDENCE,
        },
        "secondary_review_recommended": confidence_band == "intermediate_review",
    }

    if stage not in {1, 2, 3, 4, 5, 6}:
        return {
            **base,
            "growth_phase": "uncertain",
            "growth_phase_label": "Growth phase uncertain",
            "growth_velocity_descriptor": "Not assigned; predicted stage is unavailable.",
            "phase_summary": f"Stage review required. Top candidate scores: {score_breakdown_text()}",
            "estimated_time_to_skeletal_maturity": "Not estimated from this image.",
            "treatment_timing_guidance": "Obtain independent orthodontic review and confirm maturation with appropriate clinical records.",
            "treat_now_or_defer": "No automated now/defer decision; confirm the stage before timing treatment.",
            "potential_benefits": "Independent staging can reduce the risk of acting on a model error.",
            "risks_and_uncertainties": "Softmax outputs are uncalibrated model scores, not validated probabilities of correct clinical staging.",
        }

    stage_phase, velocity_descriptor = phase_for(stage)
    if confidence_band == "intermediate_review":
        phases = list(dict.fromkeys(item["growth_velocity_descriptor"] for item in candidate_phases))
        phase_range = " or ".join(phases) if phases else velocity_descriptor
        return {
            **base,
            "growth_phase": "secondary_review",
            "growth_phase_label": "Secondary Review Recommended",
            "growth_velocity_descriptor": f"Possible phases: {phase_range}.",
            "phase_summary": f"Intermediate top-stage score. Review both candidates: {score_breakdown_text()}",
            "estimated_time_to_skeletal_maturity": "Not estimated. Top-stage ambiguity does not support an individual growth-duration estimate.",
            "treatment_timing_guidance": "Arrange independent clinician review / re-staging before treatment-timing decisions. The top two candidates may imply different growth-phase interpretations.",
            "treat_now_or_defer": "Secondary review recommended; do not use this result alone to start, defer, or plan surgery.",
            "potential_benefits": "Reviewing the competing stage interpretations can reduce the risk of timing a growth-dependent intervention from an ambiguous result.",
            "risks_and_uncertainties": "The top-two values are raw softmax scores and are not calibrated probabilities. CVM stage alone does not provide a reliable individual countdown to maturity.",
        }

    if confidence_band == "low_raw_score":
        return {
            **base,
            "growth_phase": "uncertain",
            "growth_phase_label": "Low-confidence stage — manual review required",
            "growth_velocity_descriptor": f"Candidate descriptor only: {velocity_descriptor}.",
            "phase_summary": f"The leading-stage score is below the secondary-review band. Candidate scores: {score_breakdown_text()}",
            "estimated_time_to_skeletal_maturity": "Not estimated from this image.",
            "treatment_timing_guidance": "Obtain independent clinician staging and correlate with the full clinical record before considering treatment timing.",
            "treat_now_or_defer": "No automated now/defer decision. Manual review is required.",
            "potential_benefits": "Manual review can resolve whether the image supports a meaningful growth-phase interpretation.",
            "risks_and_uncertainties": "Low raw model score; this output is not suitable for treatment-timing decisions without independent review.",
        }

    if stage in (1, 2):
        label = "Prepubertal / pre-peak phase"
        summary = "CVM stages 1–2 are generally grouped as prepubertal. Growth remains possible, but this image does not identify an individual peak date."
        estimate = "Pre-peak phase; active-growth duration varies and cannot be quantified from one image."
        timing = "Review whether growth-dependent orthopedic treatment is indicated and whether monitoring toward the circumpubertal window is appropriate. Do not defer or initiate solely from this output."
        now_defer = "Clinician decision required. Consider diagnosis, remaining growth, treatment goals, and planned reassessment; this system cannot decide treat-now versus defer."
        benefits = "If a growth-dependent approach is indicated, planning before or around the growth spurt may be relevant; confirm the phase clinically."
    elif stage in (3, 4):
        label = "Circumpubertal / peak-growth window"
        summary = "CVM stages 3–4 are generally grouped as circumpubertal; a longitudinal implant study observed peak mandibular growth between these stages."
        estimate = "Peak circumpubertal velocity window may be current; remaining active-growth duration is not quantifiable from one image."
        timing = "Promptly assess whether a growth-dependent orthopedic intervention is indicated now. This is a timing-review prompt, not an instruction to start treatment."
        now_defer = "Arrange clinician review now. Whether to treat or defer depends on the patient's diagnosis, growth trajectory, treatment feasibility, and preferences."
        benefits = "If the diagnosis is appropriate, this phase may offer a relevant growth-modification window; individual response and treatment benefit are not predicted here."
    else:
        label = "Postpubertal phase"
        summary = "CVM stages 5–6 are generally grouped as postpubertal. A single image cannot confirm that all clinically relevant growth is complete."
        estimate = "Post-pubertal deceleration; residual active growth and maturity date require clinical/serial assessment."
        timing = "Review remaining growth and treatment goals. Growth modification may be less applicable; if a significant discrepancy persists after maturity is confirmed, consider multidisciplinary orthodontic and surgical evaluation."
        now_defer = "Do not make a surgical decision from this result. Confirm growth status and discuss orthodontic, surgical, and observation options with the treating team."
        benefits = "Confirming maturity can help the team discuss definitive treatment options and avoid assuming that growth modification remains effective."

    return {
        **base,
        "growth_phase": stage_phase,
        "growth_phase_label": label,
        "growth_velocity_descriptor": velocity_descriptor,
        "phase_summary": summary,
        "estimated_time_to_skeletal_maturity": estimate,
        "treatment_timing_guidance": timing,
        "treat_now_or_defer": now_defer,
        "potential_benefits": benefits,
        "risks_and_uncertainties": "CVM staging is an indirect maturity indicator; this rule is not validated for patient-specific treatment outcomes. Confirm with full clinical assessment and, when appropriate, serial records.",
        "secondary_review_recommended": False,
    }


def _make_report(image_bytes: bytes, result: dict, filename: str) -> bytes:
    """Create a hospital-style research dossier PDF with explicit limitations."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("ReportLab is required. Install backend/requirements.txt.") from exc

    try:
        with PILImage.open(BytesIO(image_bytes)) as uploaded_image:
            image_size = f"{uploaded_image.width} × {uploaded_image.height} px"
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("Uploaded file is not a readable image.") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=17 * mm, leftMargin=17 * mm,
        topMargin=15 * mm, bottomMargin=18 * mm,
        title="CLEFTSKEL-AI Decision Support Research Report",
    )
    navy = colors.HexColor("#17253f")
    teal = colors.HexColor("#137c8b")
    pale = colors.HexColor("#f2f6f9")
    amber = colors.HexColor("#fff6df")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="DossierTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19, leading=23, textColor=navy, alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle(name="DossierSub", parent=styles["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#65748b")))
    styles.add(ParagraphStyle(name="DossierSection", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, leading=14, textColor=teal, spaceBefore=8, spaceAfter=5))
    styles.add(ParagraphStyle(name="DossierBody", parent=styles["BodyText"], fontSize=9, leading=13, textColor=navy, spaceAfter=3))
    styles.add(ParagraphStyle(name="DossierSmall", parent=styles["BodyText"], fontSize=7.7, leading=10, textColor=colors.HexColor("#52627a")))
    styles.add(ParagraphStyle(name="DossierWarning", parent=styles["BodyText"], fontSize=8, leading=11, textColor=colors.HexColor("#684b16")))

    timestamp = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
    guidance = result.get("clinical_guidance") or build_growth_guidance(result)
    age = result.get("predicted_skeletal_age_years")
    age_source = result.get("age_target_source")
    age_text = f"{float(age):.1f} years" if age is not None else "Unavailable"
    if age_source == "synthetic_stage_mapping":
        age_text += " — synthetic stage-mapped proxy"
    confidence = float(result.get("cvm_stage_confidence") or 0.0) * 100
    stage_label = escape(str(result.get("cvm_stage_label", "Unknown")))
    top2_text = "; ".join(
        f"{escape(str(item.get('label', 'CVM stage')))}: {float(item.get('softmax_score_percent', 0)):.1f}%"
        for item in (result.get("cvm_stage_top2") or [])[:2]
    ) or "Not returned"

    def para(value: object, style: str = "DossierBody") -> Paragraph:
        return Paragraph(escape(str(value)), styles[style])

    def section_table(rows: list[list[object]]) -> Table:
        table = Table([[para(k, "DossierSmall"), para(v)] for k, v in rows], colWidths=[48 * mm, 122 * mm], hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), pale),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#dce4ec")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce4ec")),
        ]))
        return table

    story = [
        para(HOSPITAL_NAME, "DossierSub"),
        para("Orthodontic Growth Assessment", "DossierTitle"),
        para(f"CLEFTSKEL-AI · Research decision-support dossier · Generated {timestamp}", "DossierSub"),
        Spacer(1, 4 * mm),
        para("Patient / Image Metadata", "DossierSection"),
        section_table([
            ("Patient identifier", "Not supplied"),
            ("Image filename", filename or "Uploaded image"),
            ("Image dimensions", image_size),
            ("Image size", f"{len(image_bytes) / (1024 * 1024):.2f} MB"),
            ("Analysis timestamp", timestamp),
        ]),
        para("AI Diagnostic Findings", "DossierSection"),
        Image(BytesIO(image_bytes), width=112 * mm, height=62 * mm, kind="proportional", hAlign="CENTER"),
        section_table([
            ("Predicted CVM stage", stage_label),
            ("Top 2 stage scores", f"{top2_text} (raw softmax scores; uncalibrated)"),
            ("Leading-stage score", f"{confidence:.1f}% (not calibrated clinical certainty)"),
            ("Predicted skeletal age", age_text),
            ("ROI preprocessing", result.get("roi_status", "not reported")),
        ]),
        para("Growth Phase Assessment", "DossierSection"),
        section_table([
            ("Growth phase", guidance["growth_phase_label"]),
            ("Growth-velocity terminology", guidance.get("growth_velocity_descriptor", "Not assigned")),
            ("Interpretation", guidance["phase_summary"]),
            ("Approximate maturity timeline", guidance["estimated_time_to_skeletal_maturity"]),
            ("Active-growth duration", guidance.get("active_growth_duration_note", "Not estimated")),
            ("Age input note", guidance["age_context"]),
        ]),
        para("Treatment Timing Guidance / Clinical Recommendations", "DossierSection"),
        section_table([
            ("Timing review", guidance["treatment_timing_guidance"]),
            ("Treat now or defer?", guidance["treat_now_or_defer"]),
            ("Potential benefit to discuss", guidance["potential_benefits"]),
            ("Risks / uncertainty", guidance["risks_and_uncertainties"]),
        ]),
        Spacer(1, 4 * mm),
    ]

    disclaimer = (
        "RESEARCH PROTOTYPE — NOT A DIAGNOSIS OR AUTONOMOUS TREATMENT RECOMMENDATION. "
        "All findings and timing guidance require validation by the treating orthodontist/surgeon using the full clinical record, "
        "patient history, examination, and appropriate follow-up. Do not initiate, defer, or plan surgery solely from this report. "
        "If the age head used stage-mapped proxy targets, the displayed age is synthetic and must not be interpreted as measured skeletal age."
    )
    disclaimer_table = Table([[Paragraph(escape(disclaimer), styles["DossierWarning"])]], colWidths=[170 * mm])
    disclaimer_table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), amber), ("BOX", (0, 0), (-1, -1), 0.7, colors.HexColor("#e7cb82")), ("LEFTPADDING", (0, 0), (-1, -1), 9), ("RIGHTPADDING", (0, 0), (-1, -1), 9), ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    story.extend([
        disclaimer_table,
        Spacer(1, 3 * mm),
        Paragraph(
            'Evidence note: CVM stages are commonly grouped as CS1–2 prepubertal, CS3–4 circumpubertal, and CS5–6 postpubertal in the <link href="https://doi.org/10.2319/111517-787.1" color="#137c8b">CVM user guide</link>. A longitudinal implant study observed peak mandibular growth between CS3 and CS4 (<link href="https://pubmed.ncbi.nlm.nih.gov/18004911/" color="#137c8b">study record</link>). A systematic review also reported methodological limitations in the evidence base (<link href="https://pmc.ncbi.nlm.nih.gov/articles/PMC8813133/" color="#137c8b">review</link>). These groupings support only qualitative prompts, not an individual maturity countdown.',
            styles["DossierSmall"],
        ),
    ])

    def add_footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#dce4ec"))
        canvas.line(17 * mm, 13 * mm, A4[0] - 17 * mm, 13 * mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#71809a"))
        canvas.drawString(17 * mm, 8 * mm, "CLEFTSKEL-AI research prototype · clinician validation required")
        canvas.drawRightString(A4[0] - 17 * mm, 8 * mm, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
    return buffer.getvalue()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Return model findings plus transparent stage-based growth guidance."""
    image_bytes = await _read_image(file)
    return await _predict_or_http_error(image_bytes)


@app.post("/download-report")
async def download_report(file: UploadFile = File(...)):
    """Re-run inference and return the enhanced clinical research dossier PDF."""
    image_bytes = await _read_image(file)
    result = await _predict_or_http_error(image_bytes)
    try:
        pdf_bytes = await run_in_threadpool(_make_report, image_bytes, result, file.filename or "uploaded image")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cleftskel-growth-assessment.pdf"'},
    )
