"""FastAPI API for CLEFTSKEL-AI research predictions and reports.

Growth timing text is a transparent stage-based research heuristic. It is not a
patient-specific treatment recommendation or a substitute for clinician review.
"""

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image as PILImage, UnidentifiedImageError

from model import predict_from_image

app = FastAPI(
    title="CLEFTSKEL-AI API",
    description="Research prototype for lateral cephalogram CVM classification.",
    version="0.4.0",
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
GUIDANCE_MIN_STAGE_CONFIDENCE = float(os.getenv("CLEFTSKEL_GUIDANCE_MIN_CONFIDENCE", "0.65"))


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


def _predict_or_http_error(image_bytes: bytes) -> dict:
    try:
        result = predict_from_image(image_bytes)
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
    age_note = (
        "Synthetic stage-mapped proxy; excluded from the timing rule."
        if age is not None and age_source == "synthetic_stage_mapping"
        else ("Model age output is supportive context only; not used as a standalone maturity cutoff." if age is not None else "No skeletal-age estimate is available.")
    )

    if confidence < GUIDANCE_MIN_STAGE_CONFIDENCE or stage not in {1, 2, 3, 4, 5, 6}:
        return {
            "rules_version": "cvm-growth-window-0.1",
            "growth_phase": "uncertain",
            "growth_phase_label": "Growth phase uncertain",
            "phase_summary": "The predicted stage is missing or below the configured confidence threshold; do not use it to time an intervention.",
            "estimated_time_to_skeletal_maturity": "Not estimated from this image.",
            "treatment_timing_guidance": "Obtain orthodontist review and confirm maturation using appropriate clinical records and follow-up.",
            "treat_now_or_defer": "No automated now/defer decision. Reassess after clinician confirmation.",
            "potential_benefits": "A second review can reduce the chance of timing a growth-dependent treatment from an uncertain stage.",
            "risks_and_uncertainties": "The classifier confidence is a model score, not a calibrated probability of correct clinical staging.",
            "clinician_validation_required": True,
            "age_context": age_note,
        }

    if stage in (1, 2):
        label = "Prepubertal / pre-peak phase"
        summary = "CVM stages 1–2 are generally grouped as prepubertal. Growth remains possible, but this image does not identify an individual peak date."
        estimate = "Pre-peak phase; no patient-specific years/months countdown to maturity can be inferred."
        timing = "Review whether growth-dependent orthopedic treatment is indicated and whether monitoring toward the circumpubertal window is appropriate. Do not defer or initiate solely from this output."
        now_defer = "Clinician decision required. Consider diagnosis, remaining growth, treatment goals, and planned reassessment; this system cannot decide treat-now versus defer."
        benefits = "If a growth-dependent approach is indicated, planning before or around the growth spurt may be relevant; confirm the phase clinically."
    elif stage in (3, 4):
        label = "Circumpubertal / peak-growth window"
        summary = "CVM stages 3–4 are generally grouped as circumpubertal; a longitudinal implant study observed peak mandibular growth between these stages."
        estimate = "Peak-growth interval may be current or near; remaining time to skeletal maturity is not quantifiable from one stage."
        timing = "Promptly assess whether a growth-dependent orthopedic intervention is indicated now. This is a timing-review prompt, not an instruction to start treatment."
        now_defer = "Arrange clinician review now. Whether to treat or defer depends on the patient's diagnosis, growth trajectory, treatment feasibility, and preferences."
        benefits = "If the diagnosis is appropriate, this phase may offer a relevant growth-modification window; individual response and treatment benefit are not predicted here."
    else:
        label = "Postpubertal phase"
        summary = "CVM stages 5–6 are generally grouped as postpubertal. A single image cannot confirm that all clinically relevant growth is complete."
        estimate = "Post-peak phase; time to completed skeletal maturity is not established by this single image."
        timing = "Review remaining growth and treatment goals. Growth modification may be less applicable; if a significant discrepancy persists after maturity is confirmed, consider multidisciplinary orthodontic and surgical evaluation."
        now_defer = "Do not make a surgical decision from this result. Confirm growth status and discuss orthodontic, surgical, and observation options with the treating team."
        benefits = "Confirming maturity can help the team discuss definitive treatment options and avoid assuming that growth modification remains effective."

    return {
        "rules_version": "cvm-growth-window-0.1",
        "growth_phase": "pre_peak" if stage < 3 else ("circumpubertal_peak" if stage < 5 else "postpubertal"),
        "growth_phase_label": label,
        "phase_summary": summary,
        "estimated_time_to_skeletal_maturity": estimate,
        "treatment_timing_guidance": timing,
        "treat_now_or_defer": now_defer,
        "potential_benefits": benefits,
        "risks_and_uncertainties": "CVM staging is an indirect maturity indicator; this rule is not validated for patient-specific treatment outcomes. Confirm with full clinical assessment and, when appropriate, serial records.",
        "clinician_validation_required": True,
        "stage_confidence_threshold": GUIDANCE_MIN_STAGE_CONFIDENCE,
        "age_context": age_note,
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
            ("Stage confidence", f"{confidence:.1f}% (model score; not calibrated clinical certainty)"),
            ("Predicted skeletal age", age_text),
            ("ROI preprocessing", result.get("roi_status", "not reported")),
        ]),
        para("Growth Phase Assessment", "DossierSection"),
        section_table([
            ("Growth phase", guidance["growth_phase_label"]),
            ("Interpretation", guidance["phase_summary"]),
            ("Approximate maturity timeline", guidance["estimated_time_to_skeletal_maturity"]),
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
    return _predict_or_http_error(image_bytes)


@app.post("/download-report")
async def download_report(file: UploadFile = File(...)):
    """Re-run inference and return the enhanced clinical research dossier PDF."""
    image_bytes = await _read_image(file)
    result = _predict_or_http_error(image_bytes)
    try:
        pdf_bytes = _make_report(image_bytes, result, file.filename or "uploaded image")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return StreamingResponse(
        BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cleftskel-growth-assessment.pdf"'},
    )
