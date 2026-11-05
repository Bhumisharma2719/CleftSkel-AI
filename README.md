# CleftSkel-AI

CleftSkel-AI is a research prototype for reviewing lateral cephalogram X-rays. It provides a FastAPI backend and a browser dashboard for CVM stage classification, optional skeletal-age regression, Grad-CAM visualization, growth-phase context, and PDF report generation.

> **Research use only.** This project is not a validated medical device. Outputs require review by a qualified clinician and must not be used alone to diagnose a patient or select, defer, or plan treatment.

## Current capabilities

- MobileNetV2 transfer-learning classifier for CVMI-1 through CVMI-6.
- Balanced CVM training weights computed from the training split and applied as per-example weights in the streaming `tf.data` pipeline.
- Fine-tuning of the final MobileNetV2 layers with Adam (`1e-4`), image augmentation, `ReduceLROnPlateau`, early stopping, and best-checkpoint saving.
- Optional skeletal-age regression head. It requires direct age labels or an explicitly supplied stage-to-age proxy map.
- Top-two CVM scores and Grad-CAM overlay in the prediction response.
- Optional C2-C4 ROI localizer, trained only when normalized ROI box annotations are supplied.
- Optional standalone CVA classifier, trained only from explicit CVA labels.
- Browser dashboard with patient metadata and group selection, recent-analysis history stored locally without images or identifiers, print support, and PDF download.
- Growth-phase guidance and UCLP/non-cleft cohort context are presented as research notes, not individual clinical decisions.
- ReportLab PDF dossier with image and model findings, metadata, context, and clinician-validation disclaimer.
- `backend/create_architecture_diagram.py` generates the architecture figure at `backend/artifacts/architecture.png`.

## Project structure

```text
CleftSkel-AI/
|-- backend/
|   |-- artifacts/                 # Local checkpoints, metrics, generated diagram (gitignored)
|   |-- create_architecture_diagram.py
|   |-- main.py                    # FastAPI endpoints and PDF dossier
|   |-- model.py                   # Dataset, training, evaluation, inference, Grad-CAM
|   `-- requirements.txt
|-- frontend/
|   `-- index.html                 # Tailwind dashboard and browser-side interactions
|-- stage_age_map.json             # Optional synthetic age targets by CVM stage
`-- README.md
```

## Run locally

From the repository root, create a virtual environment and install the backend requirements:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt
```

Start the API in one terminal:

```powershell
uvicorn main:app --reload --app-dir backend
```

The API is available at `http://127.0.0.1:8000`; interactive documentation is at `http://127.0.0.1:8000/docs`.

In a second terminal, serve the frontend:

```powershell
py -m http.server 5500 --directory frontend
```

Open `http://127.0.0.1:5500`. The frontend loads Tailwind from its CDN and therefore needs internet access for Tailwind styling.

## Dataset layout

The default dataset root is `C:\Users\Bhumi\Desktop\Aariz`. Set `CLEFTSKEL_DATASET_DIR` or pass `--dataset` to use another location. The loader expects the following layout:

```text
Aariz/
|-- train/
|   |-- Cephalograms/
|   `-- Annotations/CVM Stages/*.json
|-- valid/
|   |-- Cephalograms/
|   `-- Annotations/CVM Stages/*.json
`-- test/
    |-- Cephalograms/
    `-- Annotations/CVM Stages/*.json
```

Each CVM annotation must map to an image with the same `ceph_id` (or annotation filename stem) and contain a numeric stage from 1 to 6.

## Train the CVM model

The default training configuration uses up to 100 epochs and batch size 16. It computes balanced class weights from the training labels, fine-tunes the final 35 MobileNetV2 layers while keeping batch-normalization layers frozen, augments training images (rotation 15 degrees, horizontal/vertical shifts 0.1, zoom 0.15, horizontal flip), and applies early stopping and learning-rate reduction. The best checkpoint is selected by validation loss when a validation split is available.

Run from the project root:

```powershell
python backend\model.py --train --dataset "C:\Users\Bhumi\Desktop\Aariz" --epochs 100 --batch-size 16
```

To train the optional age head with the included synthetic stage map:

```powershell
python backend\model.py --train --dataset "C:\Users\Bhumi\Desktop\Aariz" --stage-age-map ".\stage_age_map.json" --epochs 100 --batch-size 16
```

The map contains proxy targets (stage 1: 9.0, stage 2: 10.5, stage 3: 12.0, stage 4: 13.5, stage 5: 15.0, stage 6: 16.0 years). Regression trained from these values learns the supplied mapping; it is **not** a validated estimate of an individual's skeletal age. Direct age labels can instead be supplied in a CSV with `cephalogram_id,skeletal_age_years` columns:

```powershell
python backend\model.py --train --dataset "C:\Users\Bhumi\Desktop\Aariz" --age-labels ".\age_labels.csv" --epochs 100 --batch-size 16
```

Training writes `backend/artifacts/mobilenetv2.keras` and its `.labels.json` metadata sidecar by default. Model artifacts are gitignored; keep local backups of checkpoints and metrics you need.

### Optional ROI localizer

The dataset does not provide C2-C4 bounding boxes in the expected normalized format. To train a localizer, supply a CSV with a box for each training and validation image:

```csv
cephalogram_id,x_min,y_min,x_max,y_max
sample_image_id,0.30,0.55,0.58,0.88
```

```powershell
python backend\model.py --train --roi-boxes ".\c2_c3_c4_boxes.csv" --epochs 100 --batch-size 16
```

Without complete ROI labels, the CVM model uses the full image. Verify box quality and localization behavior before interpreting any ROI-based results.

### Optional CVA classifier

CVA training requires explicit binary annotations; CVM stages or arbitrary filenames are not valid CVA labels. Provide a CSV with `cephalogram_id,cva_present` where `cva_present` is 0 (normal) or 1 (anomaly present):

```powershell
python backend\model.py --train-cva --dataset "C:\Users\Bhumi\Desktop\Aariz" --cva-labels ".\cva_labels.csv" --cva-epochs 120 --batch-size 16
```

If no explicit CVA labels are available, CVA training is skipped and the API reports the detector as unavailable. Generic image measurements are not an anomaly diagnosis or CVA probability.

## Evaluation and current checkpoint results

Evaluate the saved model on the dataset's validation and held-out test splits:

```powershell
python backend\model.py --evaluate --dataset "C:\Users\Bhumi\Desktop\Aariz"
```

The current saved evaluation snapshot is in `backend/artifacts/evaluation_metrics.json`. It reports:

| Metric | Validation | Held-out test |
|---|---:|---:|
| Samples | 127 | 126 |
| Overall accuracy | 34.65% | 34.92% |
| Macro F1 | 11.50% | 10.29% |
| Cohen's kappa | -0.0127 | -0.0127 |

Held-out test accuracy by stage (support is the number of labeled images):

| Stage | Accuracy | Support |
|---|---:|---:|
| CVMI-1 | 0.00% | 4 |
| CVMI-2 | 25.00% | 4 |
| CVMI-3 | 0.00% | 6 |
| CVMI-4 | 0.00% | 32 |
| CVMI-5 | 72.88% | 59 |
| CVMI-6 | 0.00% | 21 |

These are the metrics stored with the checkpoint currently in this workspace, not a performance guarantee for future runs. They show poor overall and per-stage performance, especially for CVMI-1, 3, 4, and 6. The model is not ready for clinical use. Re-run evaluation after every training run and report the resulting metrics with the checkpoint and dataset split used. The stage-wise `accuracy_percent` in the JSON is the proportion correct within that stage; the small supports for some stages make those percentages especially uncertain.

The metrics JSON also contains calibration diagnostics. Kappa, Brier score, and expected calibration error are descriptive model diagnostics; they do not certify clinical reliability.

The project code includes separate published reference figures (74.5% average accuracy, up to 88% skeletal-maturity accuracy, and 83% CVA detection) attributed to the cited study. Those figures belong to that study's models and cohort; they are **not** CleftSkel-AI results or achieved targets.

## API endpoints

- `GET /health` - service and model-artifact status.
- `POST /predict` - multipart image upload using the field `file`; optional form fields: `patient_name`, `patient_id`, `patient_age`, `treating_orthodontist`, and `patient_group` (`uclp` or `non_uclp`). Accepts JPEG, PNG, or WebP up to 15 MB. Requires the CVM checkpoint and metadata.
- `POST /predict-cva` - optional standalone CVA classifier hook. Requires trained CVA weights and metadata.
- `POST /download-report` - same image and optional patient fields as `/predict`; returns a PDF report.

Example requests using PowerShell `curl.exe`:

```powershell
curl.exe -X POST -F "file=@C:/path/to/cephalogram.png;type=image/png" -F "patient_group=non_uclp" http://127.0.0.1:8000/predict
curl.exe -X POST -F "file=@C:/path/to/cephalogram.png;type=image/png" -F "patient_group=non_uclp" -o .\cleftskel-report.pdf http://127.0.0.1:8000/download-report
```

The response includes CVM stage and top-two raw softmax scores, model/artifact status, Grad-CAM when generation succeeds, optional age output, research-only group context, and rule-based qualitative growth guidance. Softmax scores are uncalibrated, and cohort comparison values are contextual rather than patient-specific.

Set `CLEFTSKEL_HOSPITAL_NAME` before starting Uvicorn to change the PDF header. The frontend's recent-analysis tray stores summary results in that browser's local storage; it does not store X-ray images or patient identifiers.

## Generate the architecture figure

```powershell
python backend\create_architecture_diagram.py
```

This writes `backend/artifacts/architecture.png` at 300 DPI. If Matplotlib is not installed in the active environment, install it with `python -m pip install matplotlib`.

## Limitations and privacy

The current Aariz annotations provide CVM stages. They do not provide observed skeletal age or CVA ground truth. Stage-mapped ages are synthetic proxies; CVA output remains unavailable until explicit labels and a trained checkpoint are provided. Grad-CAM is a coarse visualization and does not establish causal reasoning or anatomically faithful attention. Growth guidance is qualitative and must not independently determine treatment timing.

The API accepts patient metadata and images for a request/report. Follow institutional privacy and security requirements before using real patient information. Do not expose the development server publicly without appropriate access controls, secure transport, storage policies, and clinical validation.
