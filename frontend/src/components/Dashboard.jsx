import React, { useState, useRef } from "react";
import {
  UploadCloud,
  FileImage,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  RefreshCw,
  Eye,
  Sliders,
  Layers,
  ArrowRight,
  ShieldCheck,
  Zap,
  Info,
  ChevronRight
} from "lucide-react";
import { predictXray } from "../services/api";

export default function Dashboard() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [withGradcam, setWithGradcam] = useState(true);
  const [activeCamView, setActiveCamView] = useState("overlay"); // "overlay", "heatmap", "original", "compare"
  const [fileDetails, setFileDetails] = useState(null);

  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    setError(null);
    setResult(null);

    if (!file) return;

    const validTypes = ["image/jpeg", "image/jpg", "image/png", "image/webp"];
    const fileExt = file.name.split(".").pop().toLowerCase();
    const isValidExt = ["jpg", "jpeg", "png", "webp"].includes(fileExt);

    if (!validTypes.includes(file.type) && !isValidExt) {
      setError("Unsupported file format. Please upload a JPG, JPEG, PNG, or WEBP X-ray image.");
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setError("File size exceeds 15 MB limit. Please select a smaller radiograph image.");
      return;
    }

    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);

    // Read image dimensions
    const img = new Image();
    img.onload = () => {
      setFileDetails({
        name: file.name,
        size: (file.size / 1024).toFixed(1) + " KB",
        dimensions: `${img.naturalWidth} × ${img.naturalHeight}`,
      });
    };
    img.src = url;
  };

  const onDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const onDragLeave = () => {
    setIsDragging(false);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    setFileDetails(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please upload an X-ray image first.");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await predictXray(selectedFile, withGradcam);
      setResult(data);
    } catch (err) {
      setError(err.message || "Analysis failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-page">
      {/* Hero Welcome Banner */}
      <section className="hero-banner">
        <div className="hero-content">
          <div className="hero-pill">
            <ShieldCheck size={14} className="hero-pill-icon" />
            <span>DenseNet121 + DG-CBAM Research Architecture</span>
          </div>
          <h1 className="hero-title">
            Deep Learning Bone Fracture Screening & Decision Support
          </h1>
          <p className="hero-desc">
            Upload musculoskeletal radiographs to assess fracture probability, model confidence,
            and visualize convolutional attention localization via Difference-Guided CBAM Grad-CAM.
          </p>
        </div>
      </section>

      {/* Research Prototype & Domain Shift Notice */}
      <div className="research-domain-notice-banner">
        <div className="notice-icon-box">
          <Info size={18} />
        </div>
        <p className="notice-text">
          <strong>Research Prototype</strong> — This system was trained and internally validated on the BFRMX dataset. Performance on images from other institutions, scanners, acquisition protocols, or datasets may differ because of domain shift. Results are not a clinical diagnosis.
        </p>
      </div>

      {/* Main Workspace Layout */}
      <div className="workspace-grid">
        {/* LEFT PANEL: Upload & Preview */}
        <section className="panel-card upload-panel">
          <div className="panel-header">
            <div className="panel-title-wrap">
              <FileImage className="panel-icon" size={20} />
              <h2>Radiograph Input</h2>
            </div>
            <span className="panel-badge">Input 224 × 224</span>
          </div>

          {/* Upload Dropzone */}
          {!previewUrl ? (
            <div
              className={`dropzone ${isDragging ? "dragging" : ""}`}
              onDragOver={onDragOver}
              onDragLeave={onDragLeave}
              onDrop={onDrop}
              onClick={() => fileInputRef.current?.click()}
              id="upload-dropzone"
            >
              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                accept="image/jpeg,image/png,image/webp"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
              <div className="dropzone-icon-circle">
                <UploadCloud size={32} />
              </div>
              <p className="dropzone-title">Click to browse or drag & drop radiograph</p>
              <p className="dropzone-hint">Supported formats: JPG, JPEG, PNG, WEBP (Max 15 MB)</p>
            </div>
          ) : (
            <div className="preview-container">
              <div className="preview-img-wrapper">
                <img src={previewUrl} alt="Uploaded X-Ray Preview" className="preview-img" />
                <div className="preview-overlay-tag">Active Scan</div>
              </div>

              {fileDetails && (
                <div className="file-meta-bar">
                  <div className="meta-item">
                    <span className="meta-label">Filename:</span>
                    <span className="meta-value truncate" title={fileDetails.name}>
                      {fileDetails.name}
                    </span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Resolution:</span>
                    <span className="meta-value">{fileDetails.dimensions}</span>
                  </div>
                  <div className="meta-item">
                    <span className="meta-label">Size:</span>
                    <span className="meta-value">{fileDetails.size}</span>
                  </div>
                </div>
              )}

              <div className="preview-actions">
                <button
                  className="btn-secondary btn-sm"
                  onClick={() => fileInputRef.current?.click()}
                  id="btn-replace"
                >
                  <RefreshCw size={14} /> Replace
                </button>
                <button
                  className="btn-danger-ghost btn-sm"
                  onClick={handleClear}
                  id="btn-clear"
                >
                  <XCircle size={14} /> Remove
                </button>
              </div>

              <input
                type="file"
                ref={fileInputRef}
                style={{ display: "none" }}
                accept="image/jpeg,image/png,image/webp"
                onChange={(e) => e.target.files?.[0] && handleFileSelect(e.target.files[0])}
              />
            </div>
          )}

          {/* Options & Action */}
          <div className="analysis-options">
            <label className="toggle-label">
              <input
                type="checkbox"
                checked={withGradcam}
                onChange={(e) => setWithGradcam(e.target.checked)}
                className="custom-checkbox"
                id="checkbox-gradcam"
              />
              <div className="toggle-text">
                <span className="toggle-title">Generate Grad-CAM Attention Heatmap</span>
                <span className="toggle-sub">Visualizes convolutional activation on target fracture class</span>
              </div>
            </label>

            <button
              className={`btn-primary btn-analyze ${loading ? "loading" : ""}`}
              onClick={handleAnalyze}
              disabled={!selectedFile || loading}
              id="btn-analyze"
            >
              {loading ? (
                <>
                  <div className="spinner"></div>
                  <span>Analyzing X-Ray...</span>
                </>
              ) : (
                <>
                  <Sparkles size={18} />
                  <span>Analyze X-Ray</span>
                  <ArrowRight size={16} />
                </>
              )}
            </button>
          </div>

          {/* Error Display */}
          {error && (
            <div className="error-banner" role="alert">
              <AlertTriangle size={18} />
              <div className="error-text">
                <strong>Analysis Warning</strong>
                <p>{error}</p>
              </div>
            </div>
          )}
        </section>

        {/* RIGHT PANEL: Results & Grad-CAM */}
        <section className="panel-card results-panel">
          <div className="panel-header">
            <div className="panel-title-wrap">
              <Zap className="panel-icon" size={20} />
              <h2>Assessment Results</h2>
            </div>
            {result && (
              <span className="status-badge-success">
                <CheckCircle2 size={13} /> Analysis Completed
              </span>
            )}
          </div>

          {!result && !loading && (
            <div className="empty-results-state">
              <div className="empty-icon-circle">
                <Sliders size={32} />
              </div>
              <h3>No Active Assessment</h3>
              <p>
                Upload an X-ray radiograph and click <strong>"Analyze X-Ray"</strong> to view
                the model prediction, confidence scores, and Grad-CAM interpretability maps.
              </p>
            </div>
          )}

          {loading && (
            <div className="loading-state">
              <div className="pulse-loader"></div>
              <h3>Running Deep Learning Inference</h3>
              <p>Passing radiograph through DenseNet121 + DG-CBAM feature extractor & computing gradients...</p>
            </div>
          )}

          {result && (
            <div className="results-content">
              {/* Main Prediction Banner */}
              <div
                className={`prediction-card ${
                  result.prediction === "FRACTURE" ? "is-fracture" : "is-normal"
                }`}
                id="prediction-result-card"
              >
                <div className="pred-left">
                  <div className="pred-status-pill">
                    {result.prediction === "FRACTURE" ? (
                      <>
                        <AlertTriangle size={14} /> Attention Flagged
                      </>
                    ) : (
                      <>
                        <CheckCircle2 size={14} /> Normal Scan
                      </>
                    )}
                  </div>
                  <span className="pred-label">Model Prediction</span>
                  <h3 className="pred-value">{result.prediction}</h3>
                </div>

                <div className="pred-right">
                  <span className="conf-label">Model Confidence</span>
                  <div className="conf-value">
                    {(result.confidence * 100).toFixed(2)}%
                  </div>
                  <span className="conf-sub">Threshold locked @ 0.50</span>
                </div>
              </div>

              {/* Inline Prediction Disclaimer */}
              <div className="prediction-inline-disclaimer">
                <ShieldCheck size={14} className="inline-disc-icon" />
                <span>Prediction generated by an investigational research model. It should not be used as a standalone medical diagnosis.</span>
              </div>

              {/* Class Probability Distribution */}
              <div className="probability-section">
                <h4 className="section-subtitle">Class Probability Distribution</h4>

                {/* Fracture Bar */}
                <div className="prob-row">
                  <div className="prob-labels">
                    <span className="prob-name">
                      <span className="color-indicator fracture-dot"></span> Fracture Probability
                    </span>
                    <span className="prob-percent">
                      {(result.fracture_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="prob-track">
                    <div
                      className="prob-fill fracture-fill"
                      style={{ width: `${Math.max(result.fracture_probability * 100, 2)}%` }}
                    ></div>
                  </div>
                </div>

                {/* No Fracture Bar */}
                <div className="prob-row">
                  <div className="prob-labels">
                    <span className="prob-name">
                      <span className="color-indicator normal-dot"></span> No Fracture Probability
                    </span>
                    <span className="prob-percent">
                      {(result.no_fracture_probability * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="prob-track">
                    <div
                      className="prob-fill normal-fill"
                      style={{ width: `${Math.max(result.no_fracture_probability * 100, 2)}%` }}
                    ></div>
                  </div>
                </div>
              </div>

              {/* Technical Model Specs Card */}
              <div className="specs-summary-grid">
                <div className="spec-card">
                  <span className="spec-label">Architecture</span>
                  <span className="spec-val">DenseNet121 + DG-CBAM</span>
                </div>
                <div className="spec-card">
                  <span className="spec-label">Input Size</span>
                  <span className="spec-val">224 × 224 × 3</span>
                </div>
                <div className="spec-card">
                  <span className="spec-label">Target Layer</span>
                  <span className="spec-val">features.norm5 (1024-ch)</span>
                </div>
              </div>

              {/* Grad-CAM Explainability Section */}
              {result.gradcam_image && (
                <div className="gradcam-section">
                  <div className="gradcam-header">
                    <div className="gradcam-title-wrap">
                      <Eye className="gradcam-icon" size={18} />
                      <h4>Model Attention Visualization (Grad-CAM)</h4>
                    </div>

                    {/* View Switcher */}
                    <div className="view-switcher">
                      <button
                        className={`view-tab ${activeCamView === "overlay" ? "active" : ""}`}
                        onClick={() => setActiveCamView("overlay")}
                      >
                        Overlay
                      </button>
                      <button
                        className={`view-tab ${activeCamView === "heatmap" ? "active" : ""}`}
                        onClick={() => setActiveCamView("heatmap")}
                      >
                        Heatmap
                      </button>
                      <button
                        className={`view-tab ${activeCamView === "original" ? "active" : ""}`}
                        onClick={() => setActiveCamView("original")}
                      >
                        Original
                      </button>
                      <button
                        className={`view-tab ${activeCamView === "compare" ? "active" : ""}`}
                        onClick={() => setActiveCamView("compare")}
                      >
                        Side-by-Side
                      </button>
                    </div>
                  </div>

                  {/* Grad-CAM Images View */}
                  <div className="gradcam-view-container">
                    {activeCamView === "overlay" && (
                      <div className="single-cam-view">
                        <img
                          src={result.gradcam_image}
                          alt="Grad-CAM Composite Overlay"
                          className="cam-display-img"
                        />
                        <span className="cam-img-caption">
                          Composite Overlay (Alpha 0.45 JET Heatmap + Radiograph)
                        </span>
                      </div>
                    )}

                    {activeCamView === "heatmap" && (
                      <div className="single-cam-view">
                        <img
                          src={result.heatmap_image}
                          alt="Grad-CAM Saliency Heatmap"
                          className="cam-display-img"
                        />
                        <span className="cam-img-caption">
                          Normalized Gradient Activation Saliency (Target: Fracture Class)
                        </span>
                      </div>
                    )}

                    {activeCamView === "original" && (
                      <div className="single-cam-view">
                        <img
                          src={result.original_image || previewUrl}
                          alt="Normalized Original Radiograph"
                          className="cam-display-img"
                        />
                        <span className="cam-img-caption">
                          Input Radiograph (Normalized 224×224 RGB)
                        </span>
                      </div>
                    )}

                    {activeCamView === "compare" && (
                      <div className="compare-cam-grid">
                        <div className="compare-item">
                          <img
                            src={result.original_image || previewUrl}
                            alt="Original Radiograph"
                            className="compare-img"
                          />
                          <span>1. Original Radiograph</span>
                        </div>
                        <div className="compare-item">
                          <img
                            src={result.heatmap_image}
                            alt="Grad-CAM Heatmap"
                            className="compare-img"
                          />
                          <span>2. Grad-CAM Heatmap</span>
                        </div>
                        <div className="compare-item">
                          <img
                            src={result.gradcam_image}
                            alt="Composite Overlay"
                            className="compare-img"
                          />
                          <span>3. Attention Overlay</span>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Interpretability Note */}
                  <div className="interpretability-note">
                    <Info size={16} className="note-icon" />
                    <p>
                      The highlighted regions indicate areas of convolutional attention that contributed
                      to the model prediction. This visualization is provided for interpretability and research
                      decision support and does not constitute a medical diagnosis.
                    </p>
                  </div>
                </div>
              )}

              {/* Reset Action */}
              <div className="reset-action-bar">
                <button className="btn-secondary" onClick={handleClear} id="btn-analyze-another">
                  <RefreshCw size={15} /> Analyze Another X-Ray
                </button>
              </div>

              {/* Mandatory Research Disclaimer */}
              <div className="research-disclaimer-box">
                <ShieldCheck size={16} className="disclaimer-icon" />
                <p>
                  <strong>Research Prototype Disclaimer:</strong> BoneGuard AI is an investigational
                  research prototype. Its predictions and attention maps are intended for research and educational
                  purposes and must not be used as a standalone clinical diagnosis or replacement for licensed
                  radiological assessment.
                </p>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
