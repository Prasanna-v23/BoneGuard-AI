import React, { useState, useEffect } from "react";
import { Cpu, Layers, GitBranch, Shield, Sparkles, CheckCircle2, Sliders, Activity, Info } from "lucide-react";
import { getModelInfo } from "../services/api";

export default function ModelInfo() {
  const [modelData, setModelData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadInfo() {
      try {
        const data = await getModelInfo();
        setModelData(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }
    loadInfo();
  }, []);

  return (
    <div className="model-info-page">
      {/* Header */}
      <section className="section-hero">
        <div className="hero-pill">
          <Cpu size={14} className="hero-pill-icon" />
          <span>Architecture & Mathematical Formulation</span>
        </div>
        <h1 className="page-title">DenseNet121 + Proposed DG-CBAM Architecture</h1>
        <p className="page-subtitle">
          In-depth technical specifications of the deep feature extractor, the proposed
          Difference-Guided Convolutional Block Attention Module, and learned parameter dynamics.
        </p>
      </section>

      {/* Grid Overview */}
      <div className="model-grid">
        {/* Architecture Spec Card */}
        <div className="info-card">
          <div className="card-header-row">
            <Layers className="card-header-icon" size={20} />
            <h3>Feature Extraction Backbone</h3>
          </div>
          <div className="spec-list">
            <div className="spec-row">
              <span className="spec-key">Backbone Architecture:</span>
              <span className="spec-val-badge">DenseNet-121</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Connectivity:</span>
              <span className="spec-val">Dense Feature Reuse across 4 DenseBlocks</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Total Parameters:</span>
              <span className="spec-val highlight">
                {modelData ? modelData.total_parameters.toLocaleString() : "7,259,406"}
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Input Resolution:</span>
              <span className="spec-val">224 × 224 × 3 (RGB)</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Classification Head:</span>
              <span className="spec-val">AdaptiveAvgPool2d (1024) → Linear (2 Classes)</span>
            </div>
          </div>
        </div>

        {/* Checkpoint & Hardware Card */}
        <div className="info-card">
          <div className="card-header-row">
            <Shield className="card-header-icon" size={20} />
            <h3>Deployment Checkpoint</h3>
          </div>
          <div className="spec-list">
            <div className="spec-row">
              <span className="spec-key">Selected Checkpoint:</span>
              <span className="spec-val-badge">Exp_E_ClassWeight</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">File Path:</span>
              <span className="spec-val mono truncate" title="06_Model_Results/DGCBAM_DomainRobustness/Exp_E_ClassWeight/best_model.pth">
                06_Model_Results/.../Exp_E_ClassWeight/best_model.pth
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Active Execution Device:</span>
              <span className="spec-val-badge device">
                {modelData?.device?.toUpperCase() || "CUDA"}
                {modelData?.gpu_name ? ` (${modelData.gpu_name})` : ""}
              </span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Target Grad-CAM Layer:</span>
              <span className="spec-val">features.norm5 (1024-channel batch norm)</span>
            </div>
            <div className="spec-row">
              <span className="spec-key">Loss Function:</span>
              <span className="spec-val">Unweighted CrossEntropyLoss [1.0, 1.0]</span>
            </div>
          </div>
        </div>
      </div>

      {/* DG-CBAM Mathematical Formulation */}
      <section className="formulation-card">
        <div className="card-header-row">
          <Sparkles className="card-header-icon" size={20} />
          <h3>Proposed DG-CBAM Attention Formulation</h3>
        </div>
        <p className="formulation-intro">
          Standard Convolutional Block Attention Modules (CBAM) sequentially apply Channel Attention (
          <code className="math-code">M_c</code>) and Spatial Attention (<code className="math-code">M_s</code>
          ) without capturing the inter-attention divergence. The proposed Difference-Guided CBAM introduces
          an explicit disagreement map <code className="math-code">D = |M_c - M_s|</code> scaled by a dynamically
          optimized learnable scalar <code className="math-code">γ</code>:
        </p>

        {/* Mathematical Formula Display */}
        <div className="math-formula-box">
          <div className="formula-main">
            <span className="formula-lhs">M_new</span>
            <span className="formula-op">=</span>
            <span className="formula-term">M_s</span>
            <span className="formula-op">+</span>
            <span className="formula-gamma">γ</span>
            <span className="formula-op">·</span>
            <span className="formula-diff">|M_c − M_s|</span>
          </div>
          <div className="formula-output">
            <span>Refined Feature Map:</span>
            <code className="formula-code">F_out = F ⊗ M_new</code>
          </div>
        </div>

        {/* Dynamic Learned Gammas */}
        <div className="gammas-container">
          <h4 className="gammas-title">Learned Gamma (γ) Parameters Across Dense Stages:</h4>
          <div className="gamma-grid">
            <div className="gamma-card">
              <span className="gamma-stage">DenseBlock 1 (256-ch)</span>
              <span className="gamma-value">
                {modelData?.learned_gammas?.gamma_stage1?.toFixed(4) || "0.1015"}
              </span>
              <span className="gamma-sub">Initial: 0.1000</span>
            </div>

            <div className="gamma-card">
              <span className="gamma-stage">DenseBlock 2 (512-ch)</span>
              <span className="gamma-value">
                {modelData?.learned_gammas?.gamma_stage2?.toFixed(4) || "0.0981"}
              </span>
              <span className="gamma-sub">Initial: 0.1000</span>
            </div>

            <div className="gamma-card">
              <span className="gamma-stage">DenseBlock 3 (1024-ch)</span>
              <span className="gamma-value">
                {modelData?.learned_gammas?.gamma_stage3?.toFixed(4) || "0.1024"}
              </span>
              <span className="gamma-sub">Initial: 0.1000</span>
            </div>

            <div className="gamma-card">
              <span className="gamma-stage">DenseBlock 4 (1024-ch)</span>
              <span className="gamma-value">
                {modelData?.learned_gammas?.gamma_stage4?.toFixed(4) || "0.1134"}
              </span>
              <span className="gamma-sub">Initial: 0.1000</span>
            </div>
          </div>

          <div className="gamma-mean-box">
            <span>Converged Mean Gamma Across Network:</span>
            <strong>{modelData?.learned_gammas?.gamma_mean?.toFixed(4) || "0.1038"}</strong>
          </div>
        </div>
      </section>

      {/* Preprocessing Pipeline */}
      <section className="info-card mt-6">
        <div className="card-header-row">
          <Sliders className="card-header-icon" size={20} />
          <h3>Inference Preprocessing Pipeline</h3>
        </div>
        <div className="pipeline-steps-grid">
          <div className="pipeline-step">
            <div className="step-num">1</div>
            <div className="step-content">
              <strong>MIME & Integrity Validation</strong>
              <p>Ensures uploaded radiograph is uncorrupted JPG/PNG/WEBP under 15 MB limit.</p>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="step-num">2</div>
            <div className="step-content">
              <strong>RGB Conversion & Bilinear Resizing</strong>
              <p>Converts grayscale or RGBA scans to 3-channel RGB and aligns resolution to 224 × 224.</p>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="step-num">3</div>
            <div className="step-content">
              <strong>Standard Normalization</strong>
              <p>Applies ImageNet channel parameters (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225]).</p>
            </div>
          </div>

          <div className="pipeline-step">
            <div className="step-num">4</div>
            <div className="step-content">
              <strong>Deterministic Evaluation</strong>
              <p>Executes forward pass in eval mode with torch.no_grad() and Grad-CAM backpropagation.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
