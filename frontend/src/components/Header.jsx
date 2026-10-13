import React from "react";
import { Activity, Shield, Cpu, Database, Info, FileText, CheckCircle2, AlertCircle } from "lucide-react";

export default function Header({ activeTab, setActiveTab, backendStatus }) {
  return (
    <header className="site-header">
      <div className="header-container">
        {/* Brand & Project Identity */}
        <div className="brand-section">
          <div className="brand-logo-badge">
            <Activity className="brand-icon" size={26} />
            <div className="pulse-ring"></div>
          </div>
          <div className="brand-titles">
            <div className="brand-main">
              <span className="brand-name">BoneGuard</span>
              <span className="brand-tag">AI</span>
              <span className="research-pill">Exp-E Verified</span>
            </div>
            <p className="brand-sub">
              Advanced Bone Fracture Assessment & Clinical Decision Support
            </p>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="header-nav">
          <button
            className={`nav-btn ${activeTab === "dashboard" ? "active" : ""}`}
            onClick={() => setActiveTab("dashboard")}
            id="nav-dashboard"
          >
            <Activity size={18} />
            <span>X-Ray Analysis</span>
          </button>

          <button
            className={`nav-btn ${activeTab === "model" ? "active" : ""}`}
            onClick={() => setActiveTab("model")}
            id="nav-model"
          >
            <Cpu size={18} />
            <span>Model & DG-CBAM</span>
          </button>

          <button
            className={`nav-btn ${activeTab === "validation" ? "active" : ""}`}
            onClick={() => setActiveTab("validation")}
            id="nav-validation"
          >
            <Database size={18} />
            <span>Validation Metrics</span>
          </button>

          <button
            className={`nav-btn ${activeTab === "about" ? "active" : ""}`}
            onClick={() => setActiveTab("about")}
            id="nav-about"
          >
            <Info size={18} />
            <span>About</span>
          </button>
        </nav>

        {/* Live Backend Status */}
        <div className="status-indicator-box">
          <div className={`status-dot ${backendStatus.healthy ? "online" : "offline"}`}></div>
          <div className="status-text-group">
            <span className="status-label">Backend</span>
            <span className="status-val">
              {backendStatus.healthy
                ? `${backendStatus.device?.toUpperCase() || "ONLINE"}`
                : "OFFLINE"}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
