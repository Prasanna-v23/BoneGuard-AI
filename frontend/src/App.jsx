import React, { useState, useEffect } from "react";
import Header from "./components/Header";
import Dashboard from "./components/Dashboard";
import ModelInfo from "./components/ModelInfo";
import Validation from "./components/Validation";
import About from "./components/About";
import Footer from "./components/Footer";
import { checkHealth } from "./services/api";
import "./App.css";

export default function App() {
  const [activeTab, setActiveTab] = useState("dashboard");
  const [backendStatus, setBackendStatus] = useState({
    healthy: false,
    device: "checking...",
    service: "",
  });

  useEffect(() => {
    async function verifyBackend() {
      try {
        const health = await checkHealth();
        setBackendStatus({
          healthy: health.status === "ok",
          device: health.device || "cpu",
          service: health.service || "BoneGuard AI",
        });
      } catch (err) {
        setBackendStatus({
          healthy: false,
          device: "offline",
          service: "BoneGuard AI",
        });
      }
    }

    verifyBackend();
    const interval = setInterval(verifyBackend, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="app-root">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        backendStatus={backendStatus}
      />

      <main className="main-content">
        {activeTab === "dashboard" && <Dashboard />}
        {activeTab === "model" && <ModelInfo />}
        {activeTab === "validation" && <Validation />}
        {activeTab === "about" && <About />}
      </main>

      <Footer />
    </div>
  );
}
