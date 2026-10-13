/**
 * BoneGuard AI - API Client Service
 * =================================
 * Handles communication with the FastAPI backend.
 */

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export async function checkHealth() {
  const response = await fetch(`${API_BASE_URL}/health`);
  if (!response.ok) {
    throw new Error(`Health check failed with status: ${response.status}`);
  }
  return response.json();
}

export async function getModelInfo() {
  const response = await fetch(`${API_BASE_URL}/model-info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch model info: ${response.status}`);
  }
  return response.json();
}

export async function getValidationInfo() {
  const response = await fetch(`${API_BASE_URL}/validation-info`);
  if (!response.ok) {
    throw new Error(`Failed to fetch validation info: ${response.status}`);
  }
  return response.json();
}

export async function predictXray(imageFile, withGradcam = true) {
  const formData = new FormData();
  formData.append("image", imageFile);

  const endpoint = withGradcam
    ? `${API_BASE_URL}/predict-with-gradcam`
    : `${API_BASE_URL}/predict`;

  const response = await fetch(endpoint, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = "Analysis failed. Please try again.";
    try {
      const errorJson = await response.json();
      if (errorJson.detail) {
        errorDetail = errorJson.detail;
      }
    } catch {
      // Use fallback errorDetail
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

export { API_BASE_URL };
