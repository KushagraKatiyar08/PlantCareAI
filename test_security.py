"""Automated security, authentication, authorization, and validation tests for PlantCare AI.

Uses an isolated temporary SQLite database and synthetic image buffers to ensure
no real user data, prediction history, or model weights are ever modified.
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
    get_predictions_for_user,
    get_stats_for_user,
    hash_password,
    init_db,
    list_all_users,
    register_user,
    save_prediction_record,
    store_private_upload,
    sync_local_to_supabase,
    verify_password,
)
from report_utils import create_pdf_report
from security_utils import (
    check_system_secrets,
    sanitize_filename,
    validate_image_upload,
)


class PasswordSecurityTests(unittest.TestCase):
    """Test PBKDF2 password hashing, salt uniqueness, and constant-time verification."""

    def test_hash_generates_distinct_salts_and_hashes(self):
        hash1, salt1 = hash_password("SecurePassword@2026")
        hash2, salt2 = hash_password("SecurePassword@2026")

        self.assertNotEqual(salt1, salt2, "Each hash invocation must produce a unique random salt")
        self.assertNotEqual(hash1, hash2, "Identical passwords with different salts must produce different hashes")

    def test_verify_password_success_and_failure(self):
        pwd = "MySecretLeafPassword!123"
        hashed, salt = hash_password(pwd)

        self.assertTrue(verify_password(pwd, hashed, salt), "Correct password must verify successfully")
        self.assertFalse(verify_password("WrongPassword!123", hashed, salt), "Wrong password must be rejected")
        self.assertFalse(verify_password("", hashed, salt), "Empty password must be rejected")


class UserAuthAndRoleTests(unittest.TestCase):
    """Test user registration, role assignment, and authentication in an isolated database."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test_plantcare.db"
        init_db(self.test_db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_default_seeding(self):
        users = list_all_users(self.test_db_path)
        usernames = [u["username"].lower() for u in users]
        self.assertIn("admin", usernames, "Default admin user should be seeded")
        self.assertIn("demo_grower", usernames, "Default demo user should be seeded")

    def test_registration_validation(self):
        # Short username
        ok, msg, _ = register_user("ab", "password123", db_path=self.test_db_path)
        self.assertFalse(ok)
        self.assertIn("between 3 and 30", msg)

        # Disallowed characters in username
        ok, msg, _ = register_user("user<script>", "password123", db_path=self.test_db_path)
        self.assertFalse(ok)
        self.assertIn("alphanumeric", msg)

        # Short password
        ok, msg, _ = register_user("farmer_joe", "123", db_path=self.test_db_path)
        self.assertFalse(ok)
        self.assertIn("at least 6 characters", msg)

        # Successful registration
        ok, msg, user = register_user("farmer_joe", "ValidPass@123", role="user", db_path=self.test_db_path)
        self.assertTrue(ok)
        self.assertEqual(user["username"], "farmer_joe")
        self.assertEqual(user["role"], "user")

        # Duplicate username (case-insensitive)
        ok_dup, msg_dup, _ = register_user("FARMER_JOE", "AnotherPass!99", db_path=self.test_db_path)
        self.assertFalse(ok_dup)
        self.assertIn("already registered", msg_dup)

    def test_authentication_workflow(self):
        register_user("orchard_tech", "TechSecret@2026", role="user", db_path=self.test_db_path)

        # Correct login
        ok, msg, payload = authenticate_user("orchard_tech", "TechSecret@2026", db_path=self.test_db_path)
        self.assertTrue(ok)
        self.assertEqual(payload["username"], "orchard_tech")
        self.assertEqual(payload["role"], "user")

        # Wrong password
        ok_bad, msg_bad, _ = authenticate_user("orchard_tech", "WrongPassword", db_path=self.test_db_path)
        self.assertFalse(ok_bad)
        self.assertIn("Invalid username or password", msg_bad)

        # Nonexistent user
        ok_ghost, msg_ghost, _ = authenticate_user("nonexistent_user", "AnyPassword", db_path=self.test_db_path)
        self.assertFalse(ok_ghost)
        self.assertIn("Invalid username or password", msg_ghost)

    def test_privilege_escalation_prevented_for_normal_registrants(self):
        ok, msg, user = register_user("unauthorized_admin", "HackPass@123", role="admin", is_admin_request=False, db_path=self.test_db_path)
        self.assertTrue(ok)
        self.assertEqual(user["role"], "user", "Normal registrants attempting role=admin must be forced to role=user")

    def test_admin_request_can_create_admin_account(self):
        ok, msg, user = register_user("real_admin", "AdminPass@123", role="admin", is_admin_request=True, db_path=self.test_db_path)
        self.assertTrue(ok)
        self.assertEqual(user["role"], "admin", "Verified admin requests must be able to assign role=admin")


class RecordOwnershipAndPrivacyTests(unittest.TestCase):
    """Test multi-tenant data isolation and role-based privacy."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_db_path = Path(self.temp_dir) / "test_plantcare.db"
        init_db(self.test_db_path)

        # Register User A and User B
        _, _, self.user_a = register_user("grower_alice", "AlicePass@123", role="user", db_path=self.test_db_path)
        _, _, self.user_b = register_user("grower_bob", "BobPass@123", role="user", db_path=self.test_db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_user_record_isolation(self):
        # Alice records 2 predictions
        save_prediction_record(self.user_a["id"], "alice_tomato.jpg", "Tomato___Early_blight", 95.5, db_path=self.test_db_path)
        save_prediction_record(self.user_a["id"], "alice_potato.jpg", "Potato___Late_blight", 88.0, db_path=self.test_db_path)

        # Bob records 1 prediction
        save_prediction_record(self.user_b["id"], "bob_corn.jpg", "Corn___Common_rust", 92.4, db_path=self.test_db_path)

        # Query as Alice
        alice_records = get_predictions_for_user(self.user_a["id"], is_admin=False, db_path=self.test_db_path)
        self.assertEqual(len(alice_records), 2)
        alice_filenames = [r["image_filename"] for r in alice_records]
        self.assertIn("alice_tomato.jpg", alice_filenames)
        self.assertNotIn("bob_corn.jpg", alice_filenames, "Alice must never see Bob's records")

        # Query as Bob
        bob_records = get_predictions_for_user(self.user_b["id"], is_admin=False, db_path=self.test_db_path)
        self.assertEqual(len(bob_records), 1)
        self.assertEqual(bob_records[0]["image_filename"], "bob_corn.jpg")
        self.assertNotIn("alice_tomato.jpg", [r["image_filename"] for r in bob_records])

        # Query as Admin
        admin_records = get_predictions_for_user(1, is_admin=True, db_path=self.test_db_path)
        all_filenames = [r["image_filename"] for r in admin_records]
        self.assertIn("alice_tomato.jpg", all_filenames)
        self.assertIn("bob_corn.jpg", all_filenames)

    def test_isolated_aggregate_stats(self):
        save_prediction_record(self.user_a["id"], "leaf1.jpg", "Tomato___Early_blight", 90.0, db_path=self.test_db_path)
        save_prediction_record(self.user_a["id"], "leaf2.jpg", "Tomato___Early_blight", 80.0, db_path=self.test_db_path)

        alice_stats = get_stats_for_user(self.user_a["id"], is_admin=False, db_path=self.test_db_path)
        self.assertEqual(alice_stats["total_analyses"], 2)
        self.assertEqual(alice_stats["distinct_classes"], 1)
        self.assertEqual(alice_stats["avg_confidence"], 85.0)

        bob_stats = get_stats_for_user(self.user_b["id"], is_admin=False, db_path=self.test_db_path)
        self.assertEqual(bob_stats["total_analyses"], 0)
        self.assertEqual(bob_stats["avg_confidence"], 0.0)


class ImageUploadValidationSecurityTests(unittest.TestCase):
    """Test multi-tier image validation against malicious uploads, path traversal, and decompression bombs."""

    def _create_synthetic_image(self, width: int, height: int, fmt: str = "JPEG") -> bytes:
        img = Image.new("RGB", (width, height), color=(34, 139, 34))
        buf = io.BytesIO()
        img.save(buf, format=fmt)
        return buf.getvalue()

    def test_valid_jpeg_and_png_accepted(self):
        jpeg_bytes = self._create_synthetic_image(300, 300, "JPEG")
        ok, err, clean_name, img = validate_image_upload(jpeg_bytes, "healthy_leaf.jpg")
        self.assertTrue(ok, f"Valid JPEG should pass: {err}")
        self.assertEqual(clean_name, "healthy_leaf.jpg")
        self.assertIsNotNone(img)

        png_bytes = self._create_synthetic_image(250, 250, "PNG")
        ok, err, clean_name, img = validate_image_upload(png_bytes, "tomato_spot.png")
        self.assertTrue(ok, f"Valid PNG should pass: {err}")
        self.assertIsNotNone(img)

    def test_filename_sanitization_prevents_path_traversal(self):
        traversal_attempts = [
            ("../../../etc/passwd", "passwd.jpg"),
            ("..\\..\\secret.key", "secret.key.jpg"),
            ("test<script>.jpg", "test_script_.jpg"),
            ("null\x00byte.png", "nullbyte.png"),
            ("", "leaf_photo.jpg"),
        ]
        for malicious_name, expected_safe in traversal_attempts:
            safe = sanitize_filename(malicious_name)
            self.assertNotIn("..", safe, f"Path traversal chars must be stripped from '{malicious_name}'")
            self.assertNotIn("/", safe)
            self.assertNotIn("\\", safe)
            self.assertTrue(safe.endswith((".jpg", ".png", ".jpeg")))

    def test_reject_empty_file(self):
        ok, err, _, _ = validate_image_upload(b"", "empty.jpg")
        self.assertFalse(ok)
        self.assertIn("empty", err.lower())

    def test_reject_script_or_executable_disguised_as_image(self):
        fake_jpeg = b"<?php echo 'malicious code'; ?>"
        ok, err, _, _ = validate_image_upload(fake_jpeg, "exploit.jpg")
        self.assertFalse(ok)
        self.assertIn("Invalid image format", err)

    def test_reject_corrupted_header(self):
        # Starts with JPEG magic bytes but has corrupted payload
        corrupted_jpeg = b"\xff\xd8\xff\xe0" + b"bad_corrupted_payload" * 50
        ok, err, _, _ = validate_image_upload(corrupted_jpeg, "corrupted.jpg")
        self.assertFalse(ok)
        self.assertIn("Corrupted or malformed", err)

    def test_reject_undersized_image(self):
        tiny_img = self._create_synthetic_image(50, 50, "JPEG")
        ok, err, _, _ = validate_image_upload(tiny_img, "tiny.jpg")
        self.assertFalse(ok)
        self.assertIn("too small", err.lower())

    def test_reject_oversized_upload_bytes(self):
        # 11 MB buffer with valid header
        oversized = b"\xff\xd8\xff" + (b"\x00" * (11 * 1024 * 1024))
        ok, err, _, _ = validate_image_upload(oversized, "huge.jpg")
        self.assertFalse(ok)
        self.assertIn("exceeds maximum size", err.lower())


class ReportSecurityTests(unittest.TestCase):
    """Test PDF report generation handles malicious XML markup safely."""

    def test_pdf_report_handles_xml_injection_cleanly(self):
        malicious_label = "Tomato___<script>alert(1)</script>"
        malicious_info = {
            "title": "Title with <font color='red'>Malicious Markup</font>",
            "description": "Description with <unclosed tag & unescaped ampersand",
            "symptoms": "Symptoms with </para></story> attempt",
            "care": ["Care tip with <b>broken <i>formatting", "Special & < > \" ' chars"],
        }
        predictions = [
            {"label": malicious_label, "score": 0.98},
            {"label": "Potato___<evil_tag>", "score": 0.02},
        ]

        # Should generate a valid PDF without raising reportlab xml.parsers.expat.ExpatError
        pdf_bytes = create_pdf_report(
            label=malicious_label,
            confidence=0.98,
            predictions=predictions,
            info=malicious_info,
        )
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"), "Report should compile into a valid PDF")


class SystemHealthAndSecretTests(unittest.TestCase):
    """Test system configuration and secret validation."""

    def test_check_system_secrets(self):
        report = check_system_secrets()
        self.assertEqual(report["status"], "Healthy")
        self.assertIn("PBKDF2", report["password_hash_algorithm"])
        self.assertGreaterEqual(report["upload_max_size_mb"], 10)

    def test_sync_local_to_supabase_offline_safe(self):
        report = sync_local_to_supabase()
        self.assertIn("success", report)
        self.assertIn("synced", report)

    def test_store_private_upload_fallback(self):
        fake_bytes = b"\xff\xd8\xff\xe0" + b"0" * 100
        ok, path_out, storage_type = store_private_upload(fake_bytes, "test_leaf.jpg", user_id=999)
        self.assertTrue(ok)
        self.assertIn("user_999", path_out)
        self.assertIn(storage_type, ("cloud", "local"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
