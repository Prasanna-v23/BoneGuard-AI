# BoneGuard AI 🦴

> **Advanced Bone Fracture Assessment and Clinical Decision Support Through Deep Learning-Based X-Ray Image Analysis**

---

## 🔬 Project Overview

**BoneGuard AI** is a deep learning-based research decision-support system designed to assist clinicians and radiologists in screening musculoskeletal X-ray radiographs for bone fractures. The system integrates:

1. **Feature Extraction Backbone**: DenseNet-121 architecture leveraging dense connectivity and multi-layer feature reuse.
2. **Attention Innovation (DG-CBAM)**: Novel *Difference-Guided Convolutional Block Attention Module* that models the difference between channel and spatial attention maps ($M_{\text{new}} = M_s + \gamma \cdot |M_c - M_s|$).
3. **Explainable AI (Grad-CAM)**: Real-time visual saliency heatmaps highlighting anatomical regions of convolutional attention on the target fracture class.
4. **Clinical Web Application**: A responsive, medical-grade web dashboard built with **React.js** and backed by a high-performance **FastAPI** server.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────┐
│               React.js Web Dashboard                   │
│  (X-Ray Upload • Grad-CAM View • Validation Analytics) │
└───────────────────────────┬────────────────────────────┘
                            │ HTTP / JSON / Multipart
                            ▼
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│     (CORS • Image Validation • In-Memory Pipeline)     │
└───────────────────────────┬────────────────────────────┘
                            │ PyTorch Tensor [1, 3, 224, 224]
                            ▼
┌────────────────────────────────────────────────────────┐
│            DenseNet121 + DG-CBAM Model                 │
│      • 4 DenseBlocks + 4 DG-CBAM Attention Layers      │
│      • Target Layer: model.features.norm5 (1024-ch)    │
│      • Checkpoint: Exp_E_ClassWeight (7.26M Params)    │
└───────────────────────────┬────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐
│     Class Probabilities   │ │     Grad-CAM Heatmap      │
│ (Fracture vs No-Fracture) │ │ (Overlay / Jet / Saliency)│
└───────────────────────────┘ └───────────────────────────┘
```

---

## 📊 Research Validation & Metrics

### 1. In-Domain Primary Test: BFRMX (N = 560)
*Zero-Data-Leakage 70/15/15 Patient Partition, Locked 0.50 Threshold*

| Metric | Score | Details |
| :--- | :---: | :--- |
| **Accuracy** | **99.64%** | 558 / 560 correctly classified |
| **Precision (PPV)** | **100.00%** | Zero False Positives ($FP = 0$) |
| **Recall / Sensitivity** | **99.12%** | 224 / 226 fractures detected ($FN = 2$) |
| **Specificity (TNR)** | **100.00%** | 334 / 334 normal scans identified |
| **F1-Score** | **99.56%** | Harmonic mean of precision & recall |
| **ROC-AUC** | **0.9996** | Area under ROC curve |
| **PR-AUC** | **0.9995** | Area under Precision-Recall curve |

**BFRMX Confusion Matrix:**
- **True Positives (TP)**: 224
- **True Negatives (TN)**: 334
- **False Positives (FP)**: 0
- **False Negatives (FN)**: 2

---

### 2. External Benchmark: FracAtlas (N = 4,068)
*Independent multi-hospital benchmark evaluated strictly for zero-leakage domain shift stress-testing.*

| Metric | Score | Note |
| :--- | :---: | :--- |
| **External Accuracy** | **49.90%** | Reflects cross-scanner domain shift |
| **Sensitivity / Recall** | **75.32%** | 537 / 713 fractures detected |
| **Precision** | **22.38%** | Dataset fracture prevalence: 17.53% |
| **Specificity** | **44.50%** | 1,493 / 3,355 normal identified |
| **ROC-AUC** | **0.6479** | External discriminative capacity |
| **False Positive Reduction** | **-447 FP** | Exp-E eliminated 447 false alarms vs baseline |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Python 3.10+** (with PyTorch and CUDA support if available)
- **Node.js 18+** & **npm**

---

### Backend Setup (FastAPI)

1. Open a terminal in the project root:
   ```bash
   cd "d:/BoneGuard AI"
   ```

2. Activate the virtual environment:
   ```powershell
   # Windows PowerShell:
   .venv\Scripts\Activate.ps1
   ```

3. Install required dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. Start the FastAPI backend server:
   ```bash
   python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
   ```

Backend server runs at: **`http://127.0.0.1:8000`**  
Interactive API Docs (Swagger): **`http://127.0.0.1:8000/docs`**

---

### Frontend Setup (React.js + Vite)

1. Open a separate terminal:
   ```bash
   cd "d:/BoneGuard AI/frontend"
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the Vite development server:
   ```bash
   npm run dev
   ```

Frontend application runs at: **`http://localhost:5173`** (or `http://127.0.0.1:5173`)

---

## 📡 API Endpoints Reference

| Method | Endpoint | Description | Request | Response |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/health` | Service health & GPU check | None | `{"status": "ok", "device": "cuda", ...}` |
| `GET` | `/model-info` | Model specifications & $\gamma$ | None | `{"model_name": "DenseNet121 + DG-CBAM", ...}` |
| `GET` | `/validation-info` | Verified BFRMX & FracAtlas metrics | None | `{"internal_validation": {...}, ...}` |
| `POST` | `/predict` | Standard fracture assessment | `multipart/form-data` (`image`) | `{"prediction": "FRACTURE", "confidence": 0.9814, ...}` |
| `POST` | `/predict-with-gradcam` | Assessment + Base64 Grad-CAM | `multipart/form-data` (`image`) | `{"prediction": "FRACTURE", "gradcam_image": "data:image/png;base64,...", ...}` |

---

## 📁 Checkpoint Location

The verified Exp-E deployment checkpoint is stored at:
```text
06_Model_Results/DGCBAM_DomainRobustness/Exp_E_ClassWeight/best_model.pth
```
- **File Size**: ~28.28 MB
- **Total Parameters**: 7,259,406
- **Architecture**: DenseNet121 Backbone + 4 DG-CBAM Modules + Linear Classifier Head
- **Target Grad-CAM Layer**: `model.features.norm5` (1024 channels)

---

## 🖥️ UI Navigation & Features

1. **X-Ray Analysis Dashboard**:
   - Drag & drop or browse radiograph (JPG, JPEG, PNG, WEBP).
   - Instant image preview with resolution, size, and replace/remove controls.
   - Dual-class probability distribution bars (Fracture vs No-Fracture).
   - High-contrast prediction banner with confidence score.
   - Interactive Grad-CAM attention viewer with 4 viewing modes:
     - **Overlay**: Blended colormap with radiograph.
     - **Heatmap**: Standalone JET attention map.
     - **Original**: Normalized input radiograph.
     - **Side-by-Side**: 3-panel comparative layout.
   - "Analyze Another X-Ray" instant reset.

2. **Model & DG-CBAM**:
   - Technical specifications, parameter counts, and deployment configuration.
   - Mathematical formula of the difference-guided mechanism ($M_{\text{new}} = M_s + \gamma \cdot |M_c - M_s|$).
   - Dynamic learned $\gamma$ scalar values across DenseBlocks.

3. **Validation Metrics**:
   - Full scientific audit table of in-domain BFRMX test performance.
   - Complete confusion matrix ($TP=224, TN=334, FP=0, FN=2$).
   - Dedicated external FracAtlas benchmark section with domain generalization analysis.

4. **About**:
   - Project overview, student details, and guide information.

---

## 🔬 Known Limitation: Cross-Domain Generalization

BoneGuard AI demonstrates exceptional **in-domain test performance (99.64% accuracy, 100.00% precision)** on the primary BFRMX test partition. However, when evaluated on the independent **FracAtlas external benchmark (N = 4,068)**, performance declines to **49.90% accuracy (75.32% sensitivity, 1,862 false positives)**.

**Scientific Context:**
- This performance discrepancy is a **genuine research limitation resulting from radiographic domain shift** (differences in digital detector hardware, kVp/mAs exposure protocols, sensor grid artifacts, and institution-specific tone-curve mappings).
- Controlled diagnostic testing confirmed that the inference pipeline, preprocessing, tensor normalizations, and model weights have **0.00 numerical difference** compared to the research evaluation scripts.
- The drop in external accuracy represents a classic clinical machine learning challenge and is **not a software pipeline or UI bug**.
- The deployed Exp-E unweighted model successfully reduced **447 false-positive alarms** relative to standard baseline attention while retaining 75.32% sensitivity on unseen external scans.

---

## ⚠️ Research & Investigational Prototype Disclaimer

> **IMPORTANT**: BoneGuard AI is an **investigational research prototype**. Its predictions, probabilities, and visual attention maps are designed solely for academic evaluation, research study, and clinical decision-support demonstration. It **does not constitute a medical diagnosis** and must not replace professional radiological evaluation by a licensed physician.
