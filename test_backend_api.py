import os
import io
import sys
import time
import threading
import unittest
import requests
from PIL import Image

sys.path.insert(0, os.path.abspath("."))
import uvicorn
from backend.main import app
from backend.inference.pipeline import get_inference_service

SERVER_URL = "http://127.0.0.1:8000"

class TestBoneGuardAPI(unittest.TestCase):
    server_thread = None

    @classmethod
    def setUpClass(cls):
        # Start uvicorn in background thread if not already running
        def run_server():
            config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
            server = uvicorn.Server(config)
            server.run()

        # Check if already running
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=1.0)
            if r.status_code == 200:
                print("FastAPI server already running.")
        except Exception:
            cls.server_thread = threading.Thread(target=run_server, daemon=True)
            cls.server_thread.start()
            # Wait for server to start
            for _ in range(30):
                time.sleep(0.3)
                try:
                    r = requests.get(f"{SERVER_URL}/health", timeout=1.0)
                    if r.status_code == 200:
                        break
                except Exception:
                    pass

        # Create dummy valid image in memory
        img = Image.new("RGB", (250, 250), color=(128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        cls.sample_image_bytes = buf.getvalue()

        # Real test scan if available
        real_frac_path = os.path.join("04_Preprocessing", "02_Split", "test", "FRACTURE", "0._jumbo.jpeg")
        if os.path.exists(real_frac_path):
            with open(real_frac_path, "rb") as f:
                cls.real_frac_bytes = f.read()
        else:
            cls.real_frac_bytes = cls.sample_image_bytes

        real_normal_path = os.path.join("04_Preprocessing", "02_Split", "test", "NO_FRACTURE", "0._jumbo.jpeg")
        if not os.path.exists(real_normal_path):
            # Pick first file in NO_FRACTURE
            norm_dir = os.path.join("04_Preprocessing", "02_Split", "test", "NO_FRACTURE")
            if os.path.exists(norm_dir) and len(os.listdir(norm_dir)) > 0:
                real_normal_path = os.path.join(norm_dir, os.listdir(norm_dir)[0])

        if os.path.exists(real_normal_path):
            with open(real_normal_path, "rb") as f:
                cls.real_normal_bytes = f.read()
        else:
            cls.real_normal_bytes = cls.sample_image_bytes

    def test_01_health_endpoint(self):
        response = requests.get(f"{SERVER_URL}/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "BoneGuard AI")
        self.assertTrue(data["model_loaded"])
        self.assertIn("device", data)
        print(f"\n[PASS] /health: status={data['status']}, device={data['device']}")

    def test_02_model_info_endpoint(self):
        response = requests.get(f"{SERVER_URL}/model-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["model_name"], "DenseNet121 + DG-CBAM")
        self.assertEqual(data["num_classes"], 2)
        self.assertIn("FRACTURE", data["classes"])
        self.assertIn("NO_FRACTURE", data["classes"])
        self.assertGreater(data["total_parameters"], 7_000_000)
        self.assertIn("learned_gammas", data)
        print(f"\n[PASS] /model-info: model={data['model_name']}, params={data['total_parameters']:,}")

    def test_03_validation_info_endpoint(self):
        response = requests.get(f"{SERVER_URL}/validation-info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("internal_validation", data)
        self.assertIn("external_benchmark", data)
        internal = data["internal_validation"]
        self.assertEqual(internal["accuracy"], 99.64)
        self.assertEqual(internal["precision"], 100.00)
        self.assertEqual(internal["recall_sensitivity"], 99.12)
        self.assertEqual(internal["specificity"], 100.00)
        self.assertEqual(internal["confusion_matrix"]["tp"], 224)
        self.assertEqual(internal["confusion_matrix"]["tn"], 334)
        self.assertEqual(internal["confusion_matrix"]["fp"], 0)
        self.assertEqual(internal["confusion_matrix"]["fn"], 2)

        external = data["external_benchmark"]
        self.assertEqual(external["accuracy"], 49.90)
        self.assertEqual(external["recall_sensitivity"], 75.32)
        print(f"\n[PASS] /validation-info: BFRMX Acc={internal['accuracy']}%, FracAtlas Acc={external['accuracy']}%")

    def test_04_predict_endpoint_real_fracture(self):
        files = {"image": ("fracture_test.jpg", self.real_frac_bytes, "image/jpeg")}
        response = requests.post(f"{SERVER_URL}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["FRACTURE", "NO FRACTURE"])
        self.assertAlmostEqual(data["fracture_probability"] + data["no_fracture_probability"], 1.0, places=2)
        self.assertEqual(data["status"], "success")
        print(f"\n[PASS] /predict (Fracture Scan): prediction={data['prediction']}, frac_prob={data['fracture_probability']:.4f}, no_frac_prob={data['no_fracture_probability']:.4f}")

    def test_05_predict_endpoint_real_normal(self):
        files = {"image": ("normal_test.jpg", self.real_normal_bytes, "image/jpeg")}
        response = requests.post(f"{SERVER_URL}/predict", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn(data["prediction"], ["FRACTURE", "NO FRACTURE"])
        self.assertAlmostEqual(data["fracture_probability"] + data["no_fracture_probability"], 1.0, places=2)
        print(f"\n[PASS] /predict (Normal Scan): prediction={data['prediction']}, frac_prob={data['fracture_probability']:.4f}, no_frac_prob={data['no_fracture_probability']:.4f}")

    def test_06_predict_with_gradcam_endpoint(self):
        files = {"image": ("fracture_gradcam_test.jpg", self.real_frac_bytes, "image/jpeg")}
        response = requests.post(f"{SERVER_URL}/predict-with-gradcam", files=files)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("gradcam_image", data)
        self.assertIn("heatmap_image", data)
        self.assertIn("original_image", data)
        self.assertTrue(data["gradcam_image"].startswith("data:image/png;base64,"))
        self.assertTrue(data["heatmap_image"].startswith("data:image/png;base64,"))
        self.assertTrue(data["original_image"].startswith("data:image/png;base64,"))
        self.assertAlmostEqual(data["fracture_probability"] + data["no_fracture_probability"], 1.0, places=2)
        print(f"\n[PASS] /predict-with-gradcam: Grad-CAM Base64 length={len(data['gradcam_image'])}, prediction={data['prediction']}")

    def test_07_invalid_image_upload(self):
        # Text file uploaded as image
        files = {"image": ("test.txt", b"This is not a radiograph image.", "text/plain")}
        response = requests.post(f"{SERVER_URL}/predict", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Unsupported file format", data["detail"])
        print(f"\n[PASS] Invalid format rejection: {data['detail']}")

    def test_08_empty_image_upload(self):
        files = {"image": ("empty.jpg", b"", "image/jpeg")}
        response = requests.post(f"{SERVER_URL}/predict", files=files)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("empty", data["detail"])
        print(f"\n[PASS] Empty file rejection: {data['detail']}")

if __name__ == "__main__":
    unittest.main()
