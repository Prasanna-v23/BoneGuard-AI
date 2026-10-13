import React from "react";
import { Activity, ShieldCheck, Users, GraduationCap, Code, Cpu, Award, FileText } from "lucide-react";

export default function About() {
  return (
    <div className="about-page">
      {/* Hero */}
      <section className="section-hero">
        <div className="hero-pill">
          <Activity size={14} className="hero-pill-icon" />
          <span>Final Year Capstone Project</span>
        </div>
        <h1 className="page-title">About BoneGuard AI</h1>
        <p className="page-subtitle">
          “Advanced Bone Fracture Assessment and Clinical Decision Support Through Deep Learning-Based X-Ray Image Analysis”
        </p>
      </section>

      {/* Main Content Grid */}
      <div className="about-grid">
        {/* Project Abstract */}
        <div className="info-card">
          <div className="card-header-row">
            <FileText className="card-header-icon" size={20} />
            <h3>Project Summary & Objectives</h3>
          </div>
          <p className="about-text">
            BoneGuard AI is an automated decision-support framework engineered to assist clinicians
            and radiologists in the detection and localization of bone fractures from musculoskeletal X-ray
            radiographs.
          </p>
          <p className="about-text mt-3">
            By coupling a <strong>DenseNet-121</strong> deep feature reuse backbone with our proposed{" "}
            <strong>Difference-Guided Convolutional Block Attention Module (DG-CBAM)</strong> and{" "}
            <strong>Grad-CAM explainability</strong>, the system provides both quantitative probability
            estimates and visual attention maps without replacing licensed medical practitioners.
          </p>
        </div>

        {/* Project Team */}
        <div className="info-card">
          <div className="card-header-row">
            <Users className="card-header-icon" size={20} />
            <h3>Project Team & Supervision</h3>
          </div>
          
          <div className="team-list">
            <div className="team-member">
              <span className="member-name">GOWTHAM G</span>
              <span className="member-reg">Reg No: 731621205019</span>
            </div>
            <div className="team-member">
              <span className="member-name">PRAVEEN N</span>
              <span className="member-reg">Reg No: 731621205040</span>
            </div>
            <div className="team-member">
              <span className="member-name">YASHWANTH N</span>
              <span className="member-reg">Reg No: 731621205063</span>
            </div>
          </div>

          <div className="guide-box mt-4">
            <GraduationCap size={18} className="guide-icon" />
            <div>
              <span className="guide-title">Project Supervisor & Guide:</span>
              <strong className="guide-name">Mr. S. Nirmalkumar M.E.</strong>
              <span className="guide-role">Assistant Professor, Department of Information Technology</span>
            </div>
          </div>
        </div>
      </div>

      {/* Technology Stack Grid */}
      <section className="info-card mt-6">
        <div className="card-header-row">
          <Code className="card-header-icon" size={20} />
          <h3>Technology & Implementation Stack</h3>
        </div>

        <div className="tech-stack-grid">
          <div className="tech-item">
            <span className="tech-cat">Deep Learning Core</span>
            <span className="tech-name">PyTorch 2.6 + CUDA 12.4</span>
            <span className="tech-sub">DenseNet121 Backbone, Custom DG-CBAM Layer</span>
          </div>

          <div className="tech-item">
            <span className="tech-cat">Explainable AI</span>
            <span className="tech-name">Grad-CAM (PyTorch Hooks)</span>
            <span className="tech-sub">Target layer: features.norm5 (1024-ch)</span>
          </div>

          <div className="tech-item">
            <span className="tech-cat">Backend API</span>
            <span className="tech-name">Python FastAPI + Uvicorn</span>
            <span className="tech-sub">Asynchronous in-memory inference pipeline</span>
          </div>

          <div className="tech-item">
            <span className="tech-cat">Frontend Dashboard</span>
            <span className="tech-name">React.js + Vite</span>
            <span className="tech-sub">Interactive medical dark theme, Lucide icons</span>
          </div>
        </div>
      </section>

      {/* Regulatory & Safety Notice */}
      <div className="legal-disclaimer-box mt-6">
        <ShieldCheck size={24} className="legal-icon" />
        <div className="legal-text">
          <h4>Investigational Research Disclaimer</h4>
          <p>
            BoneGuard AI is an academic research prototype developed as part of an undergraduate capstone
            project. It has not undergone formal clinical trial clearance or FDA/CE-mark certification.
            Outputs generated by this system are for research and educational purposes only and must not
            be used as a primary diagnostic tool or replacement for expert clinical judgment.
          </p>
        </div>
      </div>
    </div>
  );
}
