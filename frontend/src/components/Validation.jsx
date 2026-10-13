import React, { useState, useEffect } from "react";
import { Database, CheckCircle2, AlertTriangle, BarChart3, ShieldCheck, Info, FileSpreadsheet } from "lucide-react";
import { getValidationInfo } from "../services/api";

export default function Validation() {
  const [valData, setValData] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        const data = await getValidationInfo();
        setValData(data);
      } catch (err) {
        console.error("Failed to fetch validation info:", err);
      }
    }
    loadData();
  }, []);

  return (
    <div className="validation-page">
      {/* Hero */}
      <section className="section-hero">
        <div className="hero-pill">
          <Database size={14} className="hero-pill-icon" />
          <span>Scientific Performance Evaluation</span>
        </div>
        <h1 className="page-title">Internal Validation & External Benchmark Metrics</h1>
        <p className="page-subtitle">
          Empirical assessment results across in-domain test data (BFRMX, N=560) and zero-leakage
          out-of-domain clinical benchmarking (FracAtlas, N=4,068).
        </p>
      </section>

      {/* SECTION 1: BFRMX Internal Test Set Performance */}
      <section className="validation-section">
        <div className="section-title-bar">
          <div className="title-with-badge">
            <h2>Internal Validation / Test Performance (BFRMX)</h2>
            <span className="dataset-pill primary">In-Domain Primary Test (N = 560)</span>
          </div>
          <span className="protocol-badge">Zero-Data-Leakage 70/15/15 Split</span>
        </div>

        {/* Metric Cards Grid */}
        <div className="metrics-cards-grid">
          <div className="metric-card gold">
            <span className="metric-card-label">Test Accuracy</span>
            <div className="metric-card-val">99.64%</div>
            <span className="metric-card-sub">558 / 560 correct</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Precision (PPV)</span>
            <div className="metric-card-val highlight-green">100.00%</div>
            <span className="metric-card-sub">Zero False Positives (FP = 0)</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Recall / Sensitivity</span>
            <div className="metric-card-val">99.12%</div>
            <span className="metric-card-sub">224 / 226 fractures caught</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Specificity (TNR)</span>
            <div className="metric-card-val highlight-green">100.00%</div>
            <span className="metric-card-sub">334 / 334 normal identified</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">F1-Score</span>
            <div className="metric-card-val">99.56%</div>
            <span className="metric-card-sub">Harmonic mean</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">ROC-AUC</span>
            <div className="metric-card-val highlight-cyan">0.9996</div>
            <span className="metric-card-sub">PR-AUC: 0.9995</span>
          </div>
        </div>

        {/* Confusion Matrix Display */}
        <div className="matrix-and-info-grid">
          <div className="confusion-matrix-box">
            <h4 className="matrix-title">BFRMX Confusion Matrix (Locked 0.50 Threshold)</h4>
            <div className="cm-table-container">
              <div className="cm-grid">
                <div className="cm-corner"></div>
                <div className="cm-col-hdr">Pred: FRACTURE</div>
                <div className="cm-col-hdr">Pred: NO FRACTURE</div>

                <div className="cm-row-hdr">True: FRACTURE (N=226)</div>
                <div className="cm-cell tp">
                  <span className="cm-num">224</span>
                  <span className="cm-desc">True Positives (TP)</span>
                </div>
                <div className="cm-cell fn">
                  <span className="cm-num">2</span>
                  <span className="cm-desc">False Negatives (FN)</span>
                </div>

                <div className="cm-row-hdr">True: NO FRACTURE (N=334)</div>
                <div className="cm-cell fp">
                  <span className="cm-num">0</span>
                  <span className="cm-desc">False Positives (FP)</span>
                </div>
                <div className="cm-cell tn">
                  <span className="cm-num">334</span>
                  <span className="cm-desc">True Negatives (TN)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="protocol-details-box">
            <h4 className="protocol-title">Scientific Validation Protocol</h4>
            <ul className="protocol-list">
              <li>
                <CheckCircle2 size={16} className="item-icon-green" />
                <span>
                  <strong>Strict Zero-Data Leakage:</strong> Training (2,611), Validation (560), and Internal
                  Test (560) partitions were locked prior to model training.
                </span>
              </li>
              <li>
                <CheckCircle2 size={16} className="item-icon-green" />
                <span>
                  <strong>No Post-Hoc Threshold Manipulation:</strong> The standard decision threshold (0.50)
                  was evaluated deterministically without test-set tuning.
                </span>
              </li>
              <li>
                <CheckCircle2 size={16} className="item-icon-green" />
                <span>
                  <strong>Unbiased Metric Verification:</strong> Exp-E was independently verified under controlled
                  confirmation experiments matching all pipeline criteria.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>

      {/* SECTION 2: External FracAtlas Benchmark */}
      <section className="validation-section external-section">
        <div className="section-title-bar">
          <div className="title-with-badge">
            <h2>External Generalization Benchmark — Not Used for Model Selection</h2>
            <span className="dataset-pill warning">Independent Multi-Hospital Stress Test (FracAtlas)</span>
          </div>
          <span className="protocol-badge alert">Zero-Training Overlap (N = 4,068)</span>
        </div>

        <div className="external-benchmark-note">
          <Info size={18} className="note-icon" />
          <p>
            <strong>Research Context & Sensitivity:</strong> FracAtlas is an independent multi-hospital radiograph benchmark
            used strictly as an out-of-domain generalization stress test. It was <em>never used</em> for
            training, threshold tuning, or model selection. The contrast between 99.64% in-domain test accuracy and 49.90%
            external benchmark accuracy demonstrates cross-institution <strong>domain-shift sensitivity</strong> due to
            differing radiographic hardware, detector pitch, and exposure protocols.
          </p>
        </div>

        {/* FracAtlas Metric Cards */}
        <div className="metrics-cards-grid">
          <div className="metric-card">
            <span className="metric-card-label">External Accuracy</span>
            <div className="metric-card-val">49.90%</div>
            <span className="metric-card-sub">Domain-shift benchmark</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Recall / Sensitivity</span>
            <div className="metric-card-val highlight-green">75.32%</div>
            <span className="metric-card-sub">537 / 713 fractures detected</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Precision (PPV)</span>
            <div className="metric-card-val">22.38%</div>
            <span className="metric-card-sub">Prevalence: 17.53% fracture</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">Specificity (TNR)</span>
            <div className="metric-card-val">44.50%</div>
            <span className="metric-card-sub">1,493 / 3,355 normal identified</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">F1-Score</span>
            <div className="metric-card-val">34.51%</div>
            <span className="metric-card-sub">Harmonic mean</span>
          </div>

          <div className="metric-card">
            <span className="metric-card-label">ROC-AUC / PR-AUC</span>
            <div className="metric-card-val highlight-cyan">0.6479</div>
            <span className="metric-card-sub">PR-AUC: 0.2711</span>
          </div>
        </div>

        {/* FracAtlas Confusion Matrix Display */}
        <div className="matrix-and-info-grid mt-4">
          <div className="confusion-matrix-box">
            <h4 className="matrix-title">FracAtlas Confusion Matrix (Locked 0.50 Threshold)</h4>
            <div className="cm-table-container">
              <div className="cm-grid">
                <div className="cm-corner"></div>
                <div className="cm-col-hdr">Pred: FRACTURE</div>
                <div className="cm-col-hdr">Pred: NO FRACTURE</div>

                <div className="cm-row-hdr">True: FRACTURE (N=713)</div>
                <div className="cm-cell tp">
                  <span className="cm-num">537</span>
                  <span className="cm-desc">True Positives (TP)</span>
                </div>
                <div className="cm-cell fn">
                  <span className="cm-num">176</span>
                  <span className="cm-desc">False Negatives (FN)</span>
                </div>

                <div className="cm-row-hdr">True: NO FRACTURE (N=3355)</div>
                <div className="cm-cell fp">
                  <span className="cm-num">1,862</span>
                  <span className="cm-desc">False Positives (FP)</span>
                </div>
                <div className="cm-cell tn">
                  <span className="cm-num">1,493</span>
                  <span className="cm-desc">True Negatives (TN)</span>
                </div>
              </div>
            </div>
          </div>

          <div className="protocol-details-box">
            <h4 className="protocol-title">Domain Generalization Findings</h4>
            <ul className="protocol-list">
              <li>
                <CheckCircle2 size={16} className="item-icon-green" />
                <span>
                  <strong>False Positive Reduction:</strong> Exp-E unweighted training eliminated 447 False Positives
                  compared to the baseline DG-CBAM configuration (1,862 vs 2,309).
                </span>
              </li>
              <li>
                <CheckCircle2 size={16} className="item-icon-green" />
                <span>
                  <strong>Retained Sensitivity:</strong> 75.32% of all external fracture scans (537/713) were detected
                  under zero fine-tuning.
                </span>
              </li>
              <li>
                <CheckCircle2 size={16} className="item-icon-green" />
                <span>
                  <strong>Scientific Transparency:</strong> External benchmark results are presented as empirical
                  research evidence of domain shift rather than patient-level diagnostic accuracy.
                </span>
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}
