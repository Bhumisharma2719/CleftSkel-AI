"""FastAPI API for CLEFTSKEL-AI predictions and downloadable reports."""

from datetime import datetime, timezone
from html import escape
from io import BytesIO
import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from model import predict_from_image

app = FastAPI(
    title="CLEFTSKEL-AI API",
    description="Research prototype for lateral cephalogram CVM classification.",
    version="0.3.0",
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
        return predict_from_image(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc


def _make_report(image_bytes: bytes, result: dict, filename: str) -> bytes:
    """Create a one-page PDF using the server's own prediction result."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError as exc:
        raise RuntimeError("ReportLab is required. Install backend/requirements.txt.") from exc

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4, rightMargin=18 * mm, leftMargin=18 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm, title="CLEFTSKEL-AI Analysis Report",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="ReportTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=18, leading=22, textColor=colors.HexColor("#17253f"), alignment=TA_LEFT, spaceAfter=3))
    styles.add(ParagraphStyle(name="Subtle", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#65748b")))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontSize=12, leading=15, textColor=colors.HexColor("#137c8b"), spaceBefore=9, spaceAfter=6))
    styles.add(ParagraphStyle(name="Disclaimer", parent=styles["Normal"], fontSize=8.5, leading=12, textColor=colors.HexColor("#75521a")))

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    safe_hospital = escape(HOSPITAL_NAME)
    safe_filename = escape(filename or "uploaded image")
    age = result.get("predicted_skeletal_age_years")
    age_text = f"{float(age):.1f} years" if age is not None else "Unavailable — age head not trained"
    age_source = result.get("age_target_source")
    if age_source == "synthetic_stage_mapping":
        age_text += " (synthetic mapped proxy)"
    stage = escape(result.get("cvm_stage_label", "Unknown"))
    confidence = float(result.get("cvm_stage_confidence", 0.0)) * 100

    story = [
        Paragraph(safe_hospital, styles["Subtle"]),
        Paragraph("Cephalometric Analysis Report", styles["ReportTitle"]),
        Paragraph(f"Generated {timestamp} · Image: {safe_filename}", styles["Subtle"]),
        Spacer(1, 5 * mm),
        Paragraph("Uploaded X-ray", styles["Section"]),
        Image(BytesIO(image_bytes), width=160 * mm, height=83 * mm, kind="proportional"),
        Spacer(1, 3 * mm),
        Paragraph("Model results", styles["Section"]),
    ]
    result_table = Table([
        ["CVM stage", stage],
        ["Stage confidence", f"{confidence:.1f}%"],
        ["Skeletal age", escape(age_text)],
        ["ROI preprocessing", escape(result.get("roi_status", "not reported"))],
        ["CVA anomaly assessment", "Not evaluated by this model"],
    ], colWidths=[52 * mm, 108 * mm])
    result_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f5f8fb")),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#52627a")),
        ("TEXTCOLOR", (1, 0), (1, -1), colors.HexColor("#17253f")),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, colors.HexColor("#dce4ec")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#dce4ec")),
    ]))
    story.extend([result_table, Spacer(1, 6 * mm)])
    if age_source == "synthetic_stage_mapping":
        disclaimer = "The age value is a synthetic target mapped from CVM stage for experimentation. It is not an observed age, validated skeletal-age estimate, or suitable for clinical decision-making."
    else:
        disclaimer = "Research prototype output only. This report is not a diagnosis and must not replace clinician assessment. CVM classification confidence is a model score, not a measure of clinical certainty."
    story.append(Paragraph(escape(disclaimer), styles["Disclaimer"]))
    doc.build(story)
    return buffer.getvalue()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run CVM classification and the optional age regression head."""
    image_bytes = await _read_image(file)
    return _predict_or_http_error(image_bytes)


@app.post("/download-report")
async def download_report(file: UploadFile = File(...)):
    """Re-run server inference and return a PDF containing the image and results."""
    image_bytes = await _read_image(file)
    result = _predict_or_http_error(image_bytes)
    try:
        pdf_bytes = _make_report(image_bytes, result, file.filename or "uploaded image")
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return StreamingResponse(
        BytesIO(pdf_bytes), media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="cleftskel-analysis-report.pdf"'},
    )
