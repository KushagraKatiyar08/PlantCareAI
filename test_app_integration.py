"""Comprehensive integration testing for PlantCare AI Streamlit application."""

import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

from history_utils import get_history_stats, load_history
from prediction_guard import assess_prediction
from report_utils import create_pdf_report
from disease_info import get_disease_info


class PlantCareIntegrationTests(unittest.TestCase):
    def test_default_diagnose_view_renders(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0, f"App threw exceptions: {at.exception}")
        self.assertEqual(at.session_state["nav_page"], "Diagnose")

    def test_navigation_to_history(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        nav = at.segmented_control(key="nav_page")
        nav.select("Prediction History")
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Prediction History")

    def test_navigation_to_disease_library(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        nav = at.segmented_control(key="nav_page")
        nav.select("Disease Library")
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Disease Library")

    def test_navigation_to_model_info(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        nav = at.segmented_control(key="nav_page")
        nav.select("Model & Limitations")
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Model & Limitations")

    def test_sample_leaf_selection_and_analysis_flow(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=15)
        
        # Select sample leaf
        sample_selector = at.selectbox(key="sample_selector")
        sample_selector.select("Potato — Early blight")
        at.run(timeout=15)
        self.assertEqual(len(at.exception), 0)
        self.assertIsNotNone(at.session_state["current_image"])
        self.assertEqual(at.session_state["sample_selected"], "Potato — Early blight")

        # Click Analyze Leaf button
        analyze_btn = None
        for btn in at.button:
            if "Analyze Leaf" in str(btn.label):
                analyze_btn = btn
                break
        
        self.assertIsNotNone(analyze_btn, "Analyze Leaf button should be present when image is loaded")
        analyze_btn.click()
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)

        # Verify prediction stored in session state
        pred = at.session_state["current_prediction"]
        self.assertIsNotNone(pred, "Prediction should be computed and stored in session state")
        self.assertIn("label", pred)
        self.assertIn("confidence", pred)
        self.assertIn("predictions", pred)
        self.assertIn("assessment", pred)
        self.assertIn("info", pred)
        self.assertIn("pdf_bytes", pred)
        self.assertGreater(len(pred["pdf_bytes"]), 1000, "PDF should be generated")

    def test_pdf_report_generation(self):
        info = get_disease_info("Potato___Early_blight")
        predictions = [
            {"label": "Potato___Early_blight", "score": 0.99},
            {"label": "Potato___Late_blight", "score": 0.01},
        ]
        pdf_bytes = create_pdf_report(
            label="Potato___Early_blight",
            confidence=0.99,
            predictions=predictions,
            info=info,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Generated output should be a valid PDF file")


    def test_admin_panel_access_when_logged_in_as_admin(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        # Simulate admin session
        at.session_state["user"] = {
            "id": 1,
            "username": "admin",
            "role": "admin",
        }
        at.run(timeout=30)
        nav = at.segmented_control(key="nav_page")
        self.assertIn("Admin Panel", nav.options)
        nav.select("Admin Panel")
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Admin Panel")

    def test_admin_panel_blocked_when_not_admin(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        # Normal user session
        at.session_state["user"] = {
            "id": 2,
            "username": "demo_grower",
            "role": "user",
        }
        at.run(timeout=30)
        nav = at.segmented_control(key="nav_page")
        self.assertNotIn("Admin Panel", nav.options)

    def test_navigation_to_sign_in_page(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        btn = at.button(key="btn_nav_signin")
        btn.click()
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Sign In")

    def test_navigation_to_sign_up_page(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        btn = at.button(key="btn_nav_signup")
        btn.click()
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Sign In")
        self.assertEqual(at.session_state["pending_auth_tab"], "Create Account")

    def test_standalone_sign_in_page_renders(self):
        at = AppTest.from_file("pages/1_Sign_In.py")
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Sign in page threw exceptions: {at.exception}")

    def test_standalone_admin_panel_blocked_for_guests(self):
        at = AppTest.from_file("pages/2_Admin_Panel.py")
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Admin panel threw exceptions on guest access: {at.exception}")

    def test_standalone_admin_panel_allowed_for_admin(self):
        at = AppTest.from_file("pages/2_Admin_Panel.py")
        at.session_state["user"] = {"id": 1, "username": "admin", "role": "admin"}
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0, f"Admin panel threw exceptions on admin access: {at.exception}")
        self.assertIsNotNone(at.button(key="standalone_btn_start_model_training"))

    def test_grower_unknown_disease_knowledge_hub(self):
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertIsNotNone(at.form(key="grower_unknown_logger_form"))

    def test_all_38_known_diseases_loaded_in_catalog(self):
        """Verify all 38 known PlantVillage / PlantCare AI classes exist in DISEASE_INFO."""
        from disease_info import DISEASE_INFO, get_disease_thumbnail_base64
        self.assertGreaterEqual(len(DISEASE_INFO), 38)

        canonical_38 = [
            "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
            "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
            "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
            "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
            "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
            "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
            "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
            "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
            "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
            "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
            "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
            "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
            "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
        ]
        for key in canonical_38:
            self.assertIn(key, DISEASE_INFO, f"Missing canonical disease key: {key}")
            entry = DISEASE_INFO[key]
            self.assertIn("title", entry)
            self.assertIn("description", entry)
            self.assertIn("symptoms", entry)
            self.assertIn("treatment", entry)
            self.assertIn("prevention", entry)
            self.assertGreater(len(entry["treatment"]), 0)
            self.assertGreater(len(entry["prevention"]), 0)

            # Ensure thumbnail base64 resolves
            thumb = get_disease_thumbnail_base64(key)
            self.assertTrue(thumb.startswith("data:image/jpeg;base64,"), f"Thumbnail failed for {key}")

    def test_disease_info_treatment_and_prevention_logic(self):
        """Verify get_disease_info returns treatment and prevention fields."""
        info = get_disease_info("Tomato___Late_blight")
        self.assertIn("treatment", info)
        self.assertIn("prevention", info)
        self.assertIn("symptoms", info)
        self.assertIn("description", info)
        self.assertEqual(info["crop"], "Tomato")
        self.assertIn("Curzate", " ".join(info["treatment"]))

    def test_disease_library_view_renders_cleanly(self):
        """Verify Disease Library view renders all conditions without exceptions."""
        at = AppTest.from_file("app.py")
        at.run(timeout=30)
        nav = at.segmented_control(key="nav_page")
        nav.select("Disease Library")
        at.run(timeout=30)
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(at.session_state["nav_page"], "Disease Library")


if __name__ == "__main__":
    unittest.main(verbosity=2)
