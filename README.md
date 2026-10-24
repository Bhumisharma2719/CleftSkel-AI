# CLEFTSKEL-AI

FastAPI and HTML/Tailwind capstone application for research into lateral cephalogram CVM classification.

## Project structure

```text
CLEFTSKEL-AI/
├── backend/
│   ├── main.py
│   ├── model.py
│   └── requirements.txt
├── frontend/
│   └── index.html
└── README.md
```

## Run locally

### 1. Start the backend

From the project root, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
```

Start FastAPI:

```powershell
uvicorn main:app --reload --app-dir backend
```

The API runs at `http://127.0.0.1:8000`; interactive API docs are at `http://127.0.0.1:8000/docs`.

### 2. Start the frontend

Open a second terminal at the project root:

```powershell
py -m http.server 5500 --directory frontend
```

Visit `http://127.0.0.1:5500`. The page uses Tailwind's CDN, so an internet connection is needed to load its styling.

## Train the MobileNetV2 model

The default dataset path is `C:\Users\Bhumi\Desktop\Aariz`. You can override it with `--dataset` or `CLEFTSKEL_DATASET_DIR`. The loader reads `Cephalograms` plus `Annotations/CVM Stages` from the `train`, `valid`, and `test` splits. After installing the expanded requirements, run from the project root:

```powershell
python backend\model.py --train --epochs 15 --batch-size 16
```

The trained model and stage metadata are written to `backend/artifacts/`. Start the API as before; `/predict` loads this model on its first request and returns the CVM stage, confidence, and (if trained) skeletal age.

**Label limitation:** the Aariz annotations identify cervical vertebral maturation (CVM) stages; they do not contain skeletal age in years or labels for cervical vertebral anomalies (CVA). The CVM classifier trains from Aariz. CVA anomaly detection and age output remain unavailable until their respective ground-truth labels are provided. To train the age regression head, provide a CSV containing `cephalogram_id,skeletal_age_years` columns:

```powershell
python backend\model.py --train --age-labels path\to\age_labels.csv
```

The age head is trained from explicit age CSV values when all train and validation records have labels. For an experiment with discrete stage labels only, you can opt into a synthetic map JSON such as:

```json
{"1": 9.0, "2": 10.5, "3": 12.0, "4": 13.5, "5": 15.0, "6": 16.0}
```

Those numbers are user supplied placeholders, not validated clinical targets. Run `python backend\model.py --train --stage-age-map path\to\stage_age_map.json`. Reports and API metadata identify this output as a synthetic mapped proxy. Do not use it as a true age estimate or for care decisions. An age head trained with real, patient-linked age labels is needed to make age claims.

## Downloadable PDF report

After successful image analysis, use **Download PDF Report**. `POST /download-report` runs inference on the uploaded file and returns a one-page PDF with the X-ray, CVM stage, model confidence, age output/source, configurable hospital name, and UTC timestamp. Set the `CLEFTSKEL_HOSPITAL_NAME` environment variable before starting Uvicorn to change the report header. The report states that CVA anomaly detection is not part of the current model.

## API

- `GET /health` — readiness check.
- `POST /predict` — multipart form upload using the field name `file`; accepts JPEG, PNG, or WebP up to 15 MB. Returns HTTP 503 until a trained model and metadata are present.
- `POST /download-report` — same multipart upload; returns a PDF attachment.

This starter is not a medical device and has not been validated for clinical use. Handle patient images according to your institution's privacy and security requirements before deployment.
