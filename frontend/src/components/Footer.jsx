import React from "react";
import { Activity, Shield } from "lucide-react";

export default function Footer() {
  return (
    <footer className="site-footer">
      <div className="footer-container">
        <div className="footer-left">
          <div className="footer-brand">
            <Activity size={18} className="footer-icon" />
            <span>BoneGuard AI</span>
          </div>
          <p className="footer-subtitle">
            Advanced Bone Fracture Assessment & Clinical Decision Support Through Deep Learning-Based X-Ray Image Analysis
          </p>
        </div>

        <div className="footer-right">
          <div className="footer-disclaimer-pill">
            <Shield size={14} />
            <span>Investigational Research Prototype — Not for Clinical Diagnosis</span>
          </div>
          <span className="footer-copy">
            © 2026 BoneGuard AI Research Project • Department of Information Technology
          </span>
        </div>
      </div>
    </footer>
  );
}
