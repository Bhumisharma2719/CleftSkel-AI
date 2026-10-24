"""FastAPI entry point for CLEFTSKEL-AI."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from model import predict_from_image

app = FastAPI(
    title="CLEFTSKEL-AI API",
    description="Lateral cephalogram CVM stage analysis API.",
    version="0.2.0",
)

# Local frontend development server. Restrict this list to your deployed frontend
# origin when you deploy the application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500", "http://127.0.0.1:8000", "http://localhost:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_FILE_BYTES = 15 * 1024 * 1024


@app.get("/health")
async def health():
    return {"status": "ok", "service": "CLEFTSKEL-AI API"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Run CVM classification (and age regression when trained) on an X-ray."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP image.")

    image_bytes = await file.read(MAX_FILE_BYTES + 1)
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB limit.")

    try:
        result = predict_from_image(image_bytes)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        # Avoid returning internal model details to API clients.
        raise HTTPException(status_code=500, detail="Image analysis failed.") from exc

    return {
        "predicted_skeletal_age_years": result["predicted_skeletal_age_years"],
        "cvm_stage": result["cvm_stage"],
        "cvm_stage_label": result["cvm_stage_label"],
        "cva_status": result["cva_status"],
        "cvm_stage_confidence": result["cvm_stage_confidence"],
        "status": result["status"],
        "message": result["message"],
    }
