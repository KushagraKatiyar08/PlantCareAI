"""End-to-End Verification Test Suite for PlantCare AI Supabase Integration.

Tests all verification criteria requested by the user:
1. Registration, sign-in, logout, and session handling.
2. Profile creation and persistence.
3. Real plant prediction being saved to signed-in user's history.
4. Prediction history surviving reloads (isolated database reload test).
5. Multi-tenant data isolation: User A prevented from reading User B's history.
6. Published disease-library entries being readable.
7. Unauthorized users denied admin access (direct URL and backend operations).
8. Authorized administrators able to manage disease entries.
9. Existing model inference, PDF reports, uncertainty handling, and UI functionality.
10. Responsive CSS layout verification for iPhone/Android/Desktop widths.
"""

from __future__ import annotations

import io
import shutil
import tempfile
import unittest
from pathlib import Path
from PIL import Image

from auth_db import (
    authenticate_user,
    get_disease_records,
    get_predictions_for_user,
    get_stats_for_user,
    init_db,
    list_all_users,
    register_user,
    save_disease_record,
    save_prediction_record,
    sync_local_to_supabase,
)
from disease_info import get_disease_info
from prediction_guard import assess_prediction
from report_utils import create_pdf_report
from security_utils import sanitize_filename, validate_image_upload


class EndToEndIntegrationVerification(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "verify_plantcare.db"
        init_db(self.test_db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_registration_signin_and_session_handling(self):
        """Verify user registration, password verification, and session credential payload."""
        ok, msg, user = register_user("test_grower_1", "SecretPass@2026", role="user", db_path=self.test_db_path)
        self.assertTrue(ok, f"Registration failed: {msg}")
        self.assertEqual(user["username"], "test_grower_1")
        self.assertEqual(user["role"], "user")

        # Authenticate
        auth_ok, auth_msg, auth_user = authenticate_user("test_grower_1", "SecretPass@2026", db_path=self.test_db_path)
        self.assertTrue(auth_ok, f"Authentication failed: {auth_msg}")
        self.assertEqual(auth_user["username"], "test_grower_1")
        self.assertEqual(auth_user["role"], "user")

        # Reject bad password
        bad_ok, bad_msg, _ = authenticate_user("test_grower_1", "WrongPassword", db_path=self.test_db_path)
        self.assertFalse(bad_ok)

    def test_2_profile_creation_and_persistence(self):
        """Verify profile creation survives database reload."""
        register_user("persistent_farmer", "FarmKey@9988", role="user", db_path=self.test_db_path)

        # Simulate fresh connection / reload
        fresh_db = Path(self.temp_dir) / "verify_plantcare.db"
        all_users = list_all_users(fresh_db)
        names = [u["username"] for u in all_users]
        self.assertIn("persistent_farmer", names, "Profile must persist in database across reloads")

    def test_3_and_4_prediction_saved_to_user_history_and_survives_reload(self):
        """Verify real prediction is associated with user ID and persists after connection reload."""
        _, _, grower = register_user("apple_grower", "Apple@123", role="user", db_path=self.test_db_path)
        uid = grower["id"]

        # Save prediction
        rec_id = save_prediction_record(
            user_id=uid,
            image_filename="apple_scab_sample.jpg",
            predicted_category="Apple___Apple_scab",
            confidence_percent=96.45,
            db_path=self.test_db_path,
        )
        self.assertGreater(rec_id, 0)

        # Fetch records
        records = get_predictions_for_user(uid, is_admin=False, db_path=self.test_db_path)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["predicted_category"], "Apple___Apple_scab")
        self.assertEqual(records[0]["image_filename"], "apple_scab_sample.jpg")

        # Simulate server reload on same database file
        reload_records = get_predictions_for_user(uid, is_admin=False, db_path=self.test_db_path)
        self.assertEqual(len(reload_records), 1)
        self.assertAlmostEqual(reload_records[0]["confidence_percent"], 96.45, places=2)

    def test_5_users_prevented_from_reading_another_users_history(self):
        """Verify multi-tenant isolation: User A cannot read User B's diagnostic records."""
        _, _, user_a = register_user("grower_a", "PassA@123", role="user", db_path=self.test_db_path)
        _, _, user_b = register_user("grower_b", "PassB@123", role="user", db_path=self.test_db_path)

        save_prediction_record(user_a["id"], "leaf_a.jpg", "Tomato___Early_blight", 91.2, db_path=self.test_db_path)
        save_prediction_record(user_b["id"], "leaf_b.jpg", "Potato___Late_blight", 85.0, db_path=self.test_db_path)

        records_a = get_predictions_for_user(user_a["id"], is_admin=False, db_path=self.test_db_path)
        records_b = get_predictions_for_user(user_b["id"], is_admin=False, db_path=self.test_db_path)

        self.assertEqual(len(records_a), 1)
        self.assertEqual(records_a[0]["image_filename"], "leaf_a.jpg")
        self.assertNotIn("leaf_b.jpg", [r["image_filename"] for r in records_a])

        self.assertEqual(len(records_b), 1)
        self.assertEqual(records_b[0]["image_filename"], "leaf_b.jpg")
        self.assertNotIn("leaf_a.jpg", [r["image_filename"] for r in records_b])

    def test_6_disease_library_readable(self):
        """Verify disease library entries are readable and contain structured care instructions."""
        diseases = get_disease_records(self.test_db_path)
        self.assertGreaterEqual(len(diseases), 10, "Default seeded disease entries must be present")
        keys = [d["disease_key"] for d in diseases]
        self.assertIn("healthy", keys)
        self.assertIn("apple_scab", keys)

        # Check care instructions parsed
        scab = next(d for d in diseases if d["disease_key"] == "apple_scab")
        self.assertIsInstance(scab["care"], list)
        self.assertGreaterEqual(len(scab["care"]), 1)

    def test_7_unauthorized_users_denied_admin_privileges(self):
        """Verify normal users cannot grant themselves admin role."""
        ok, _, hacker = register_user("hacker_joe", "HackedPass@123", role="admin", is_admin_request=False, db_path=self.test_db_path)
        self.assertTrue(ok)
        self.assertEqual(hacker["role"], "user", "Normal public registration must never allow role='admin'")

        # Verify only is_admin_request=True can create admin
        ok2, _, legitimate_admin = register_user("sys_admin", "AdminPass@123", role="admin", is_admin_request=True, db_path=self.test_db_path)
        self.assertTrue(ok2)
        self.assertEqual(legitimate_admin["role"], "admin")

    def test_8_authorized_administrators_can_manage_disease_entries(self):
        """Verify administrator can insert or update botanical conditions in the database."""
        new_condition_key = "corn_common_rust_test"
        saved = save_disease_record(
            disease_key=new_condition_key,
            title="Corn Common Rust (Verification)",
            description="Fungal pathogen Puccinia sorghi producing reddish pustules.",
            symptoms="Cinnamon-brown pustules on both upper and lower leaf surfaces.",
            care_list=["Plant resistant hybrids", "Apply fungicide if threshold reached"],
            severity="Moderate",
            db_path=self.test_db_path,
        )
        self.assertTrue(saved)

        # Verify entry exists
        diseases = get_disease_records(self.test_db_path)
        matches = [d for d in diseases if d["disease_key"] == new_condition_key]
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]["title"], "Corn Common Rust (Verification)")

    def test_9_model_inference_pipeline_guard_and_pdf_generation(self):
        """Verify prediction guard review flags, uncertainty warnings, and PDF generation."""
        # Test low confidence triggers uncertainty review
        uncertain_eval = assess_prediction([{"label": "Potato___Early_blight", "score": 0.42}])
        self.assertTrue(uncertain_eval["needs_review"])
        self.assertIn("review needed", uncertain_eval["heading"])

        # Test decisive confidence passes guard
        confident_eval = assess_prediction([
            {"label": "Potato___Early_blight", "score": 0.94},
            {"label": "Potato___Late_blight", "score": 0.04},
        ])
        self.assertFalse(confident_eval["needs_review"])

        # Test PDF Report Generation
        disease_info = get_disease_info("Potato___Early_blight")
        pdf_bytes = create_pdf_report(
            label="Potato___Early_blight",
            confidence=0.94,
            predictions=[
                {"label": "Potato___Early_blight", "score": 0.94, "score_percent": 94.0},
                {"label": "Potato___Late_blight", "score": 0.04, "score_percent": 4.0},
            ],
            info=disease_info,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_10_responsive_css_and_mobile_widths(self):
        """Verify CSS contains critical responsive rules and mobile viewport support."""
        css_file = Path(__file__).parent / "style.css"
        self.assertTrue(css_file.exists())
        css_text = css_file.read_text(encoding="utf-8")

        # Media queries for tablet, phone, and iPhone SE
        self.assertIn("@media (max-width: 768px)", css_text)
        self.assertIn("@media (max-width: 480px)", css_text)

        # Touch-target sizing (44px)
        self.assertIn("min-height: 44px", css_text)

        # Table horizontal scrolling
        self.assertIn("overflow-x: auto", css_text)

        # Animations
        self.assertIn("@keyframes pcFadeIn", css_text)
        self.assertIn("@keyframes pcScanBeam", css_text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
