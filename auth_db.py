"""Database persistence, user authentication, and authorization for PlantCare AI.

Supports dual-mode storage:
1. Supabase PostgreSQL Cloud Database (via official supabase-py client or REST API)
2. Local thread-safe SQLite storage (plantcare.db) with automatic fallback and sync.

Stores:
- User accounts (salted PBKDF2-HMAC-SHA256 password hashing, roles)
- Prediction history with user record ownership
- Disease knowledge library
"""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import secrets
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

DB_FILE = Path(__file__).parent / "plantcare.db"
HISTORY_CSV_FILE = Path(__file__).parent / "prediction_history.csv"
PBKDF2_ITERATIONS = 100_000

# Cache for Supabase client
_SUPABASE_CLIENT = None
_SUPABASE_INITIALIZED = False


def get_supabase_client() -> Any | None:
    """Retrieve or create an active Supabase client if credentials are configured."""
    global _SUPABASE_CLIENT, _SUPABASE_INITIALIZED

    url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
        or os.environ.get("SUPABASE_KEY", "").strip()
        or os.environ.get("SUPABASE_ANON_KEY", "").strip()
    )

    # Check Streamlit secrets if running inside Streamlit
    if not (url and key):
        try:
            import streamlit as st

            if hasattr(st, "secrets"):
                url = url or st.secrets.get("SUPABASE_URL", "").strip()
                key = (
                    key
                    or st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
                    or st.secrets.get("SUPABASE_KEY", "").strip()
                    or st.secrets.get("SUPABASE_ANON_KEY", "").strip()
                )
        except Exception:
            pass

    # Also check a local .env file if available
    if not (url and key):
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            try:
                for line in env_file.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if line.startswith("SUPABASE_URL="):
                        url = line.split("=", 1)[1].strip().strip('"').strip("'")
                    elif line.startswith("SUPABASE_KEY=") or line.startswith("SUPABASE_SERVICE_ROLE_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"').strip("'")
            except Exception:
                pass

    if not (url and key):
        return None

    try:
        from supabase import create_client

        _SUPABASE_CLIENT = create_client(url, key)
        _SUPABASE_INITIALIZED = True
        return _SUPABASE_CLIENT
    except Exception:
        return None


def get_database_status() -> dict[str, Any]:
    """Inspect and return current database connectivity and sync status."""
    client = get_supabase_client()
    supabase_configured = client is not None
    supabase_connected = False
    supabase_error = None

    if supabase_configured:
        try:
            # Quick probe on remote users or diseases table
            res = client.table("users").select("id").limit(1).execute()
            supabase_connected = True
        except Exception as err:
            supabase_error = str(err)
            supabase_connected = False

    # Check local SQLite statistics
    sqlite_exists = DB_FILE.exists()
    user_count = 0
    prediction_count = 0
    disease_count = 0

    if sqlite_exists:
        try:
            with get_db_connection() as conn:
                c = conn.cursor()
                user_count = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                prediction_count = c.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
                try:
                    disease_count = c.execute("SELECT COUNT(*) FROM diseases").fetchone()[0]
                except Exception:
                    disease_count = 0
        except Exception:
            pass

    url = os.environ.get("SUPABASE_URL", "")
    masked_url = ""
    if url:
        masked_url = url[:14] + "..." + url[-10:] if len(url) > 24 else url

    return {
        "mode": "supabase" if supabase_connected else "sqlite",
        "provider": "Supabase PostgreSQL Cloud" if supabase_connected else "Local SQLite (plantcare.db)",
        "supabase_configured": supabase_configured,
        "supabase_connected": supabase_connected,
        "supabase_url": masked_url,
        "supabase_error": supabase_error,
        "local_db_exists": sqlite_exists,
        "counts": {
            "users": user_count,
            "predictions": prediction_count,
            "diseases": disease_count,
        },
    }


def _is_default_db(db_path: Path | str) -> bool:
    """Return True if db_path is the production DB_FILE, False if an isolated test DB."""
    try:
        return Path(db_path).resolve() == DB_FILE.resolve()
    except Exception:
        return str(db_path) == str(DB_FILE)


def get_db_connection(db_path: Path | str = DB_FILE) -> sqlite3.Connection:
    """Create a thread-safe connection to the SQLite database."""
    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    """Hash password using salted PBKDF2-HMAC-SHA256."""
    if salt is None:
        salt = secrets.token_hex(16)

    dk = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        PBKDF2_ITERATIONS,
    )
    return dk.hex(), salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    """Verify password against stored hash using constant-time comparison."""
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)


_INITIALIZED_DBS: set[str] = set()


def init_db(db_path: Path | str = DB_FILE, force: bool = False) -> None:
    """Initialize database tables, seed default accounts and disease records, and migrate legacy CSV records."""
    db_key = str(Path(db_path).resolve())
    if db_key in _INITIALIZED_DBS and not force:
        return

    # 1. Initialize SQLite tables
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()

        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL COLLATE NOCASE,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'user',
                created_at TEXT NOT NULL,
                last_login TEXT
            );
        """)

        # Predictions table with ownership
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                predicted_category TEXT NOT NULL,
                confidence_percent REAL NOT NULL,
                status TEXT DEFAULT 'completed',
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            );
        """)

        # Disease Library table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diseases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                disease_key TEXT UNIQUE NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                symptoms TEXT NOT NULL,
                care_json TEXT NOT NULL,
                severity TEXT NOT NULL DEFAULT 'Moderate',
                crop TEXT DEFAULT '',
                pathogen TEXT DEFAULT '',
                treatment_json TEXT DEFAULT '[]',
                prevention_json TEXT DEFAULT '[]'
            );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_diseases_key ON diseases(disease_key);")

        # Migrate existing schema if columns are missing
        cursor.execute("PRAGMA table_info(diseases);")
        existing_cols = {col[1] for col in cursor.fetchall()}
        for col_name, col_type, default_val in [
            ("crop", "TEXT", "''"),
            ("pathogen", "TEXT", "''"),
            ("treatment_json", "TEXT", "'[]'"),
            ("prevention_json", "TEXT", "'[]'"),
        ]:
            if col_name not in existing_cols:
                try:
                    cursor.execute(f"ALTER TABLE diseases ADD COLUMN {col_name} {col_type} DEFAULT {default_val};")
                except Exception:
                    pass

        # Seed default administrator and demo grower if users table is empty
        cursor.execute("SELECT COUNT(*) FROM users;")
        if cursor.fetchone()[0] == 0:
            now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            admin_hash, admin_salt = hash_password("Admin@PlantCare2026")
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, 'admin', ?);
            """, ("admin", admin_hash, admin_salt, now_iso))

            demo_hash, demo_salt = hash_password("Grower@123")
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, 'user', ?);
            """, ("demo_grower", demo_hash, demo_salt, now_iso))

        # Seed or refresh disease library from disease_info.py (ensure all 38+ classes are loaded)
        cursor.execute("SELECT COUNT(*) FROM diseases;")
        current_disease_count = cursor.fetchone()[0]
        if current_disease_count < 38:
            try:
                from disease_info import DISEASE_INFO
                for key, data in DISEASE_INFO.items():
                    title = data.get("title", key.replace("_", " ").title())
                    desc = data.get("description", "")
                    symptoms = data.get("symptoms", "")
                    treat_list = data.get("treatment", [])
                    prev_list = data.get("prevention", [])
                    care_list = data.get("care", treat_list + prev_list)
                    care_str = json.dumps(care_list)
                    treat_str = json.dumps(treat_list)
                    prev_str = json.dumps(prev_list)
                    severity = data.get("severity") or ("Low" if "healthy" in key.lower() else "High" if "blight" in key.lower() or "rot" in key.lower() else "Moderate")
                    crop = data.get("crop", "")
                    pathogen = data.get("pathogen", "")

                    cursor.execute("""
                        INSERT INTO diseases (disease_key, title, description, symptoms, care_json, severity, crop, pathogen, treatment_json, prevention_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(disease_key) DO UPDATE SET
                            title=excluded.title,
                            description=excluded.description,
                            symptoms=excluded.symptoms,
                            care_json=excluded.care_json,
                            severity=excluded.severity,
                            crop=excluded.crop,
                            pathogen=excluded.pathogen,
                            treatment_json=excluded.treatment_json,
                            prevention_json=excluded.prevention_json;
                    """, (key, title, desc, symptoms, care_str, severity, crop, pathogen, treat_str, prev_str))
            except Exception:
                pass

        # Migrate existing CSV records if predictions table is empty
        cursor.execute("SELECT COUNT(*) FROM predictions;")
        if cursor.fetchone()[0] == 0 and HISTORY_CSV_FILE.exists() and HISTORY_CSV_FILE.stat().st_size > 0:
            try:
                with HISTORY_CSV_FILE.open("r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ts = row.get("timestamp", "")
                        fname = row.get("image_filename", "")
                        cat = row.get("predicted_category", "")
                        try:
                            conf = float(row.get("confidence_percent", 0.0))
                        except (ValueError, TypeError):
                            conf = 0.0

                        if cat:
                            cursor.execute("""
                                INSERT INTO predictions (user_id, timestamp, image_filename, predicted_category, confidence_percent)
                                VALUES (1, ?, ?, ?, ?);
                            """, (ts, fname, cat, conf))
            except Exception:
                pass

        conn.commit()

    _INITIALIZED_DBS.add(db_key)

    # 2. Synchronize to Supabase if available (only on forced re-initialization or when Supabase client is already active)
    if force and _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                # Seed default users in Supabase if empty
                res = client.table("users").select("id").limit(1).execute()
                if not res.data:
                    admin_hash, admin_salt = hash_password("Admin@PlantCare2026")
                    demo_hash, demo_salt = hash_password("Grower@123")
                    client.table("users").upsert([
                        {
                            "username": "admin",
                            "password_hash": admin_hash,
                            "salt": admin_salt,
                            "role": "admin",
                        },
                        {
                            "username": "demo_grower",
                            "password_hash": demo_hash,
                            "salt": demo_salt,
                            "role": "user",
                        },
                    ]).execute()
            except Exception:
                pass


def register_user(
    username: str,
    password: str,
    role: str = "user",
    is_admin_request: bool = False,
    db_path: Path | str = DB_FILE,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Register a new user account with validated credentials in SQLite and Supabase."""
    uname = username.strip()
    if len(uname) < 3 or len(uname) > 30:
        return False, "Username must be between 3 and 30 characters.", None
    if not uname.isalnum() and "_" not in uname:
        return False, "Username can only contain alphanumeric characters and underscores.", None
    if len(password) < 6:
        return False, "Password must be at least 6 characters long.", None

    # Privilege escalation defense: only verified administrator sessions can assign the admin role
    if role.lower() == "admin" and not is_admin_request:
        assigned_role = "user"
    else:
        assigned_role = "admin" if role.lower() == "admin" else "user"

    pwd_hash, salt = hash_password(password)
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Insert into local SQLite
    user_id = None
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, salt, role, created_at)
                VALUES (?, ?, ?, ?, ?);
            """, (uname, pwd_hash, salt, assigned_role, now_iso))
            user_id = cursor.lastrowid
            conn.commit()
    except sqlite3.IntegrityError:
        return False, f"Username '{uname}' is already registered.", None
    except Exception as e:
        return False, f"Registration error: {e}", None

    # 2. Insert into Supabase if configured (only in production database mode)
    if _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                client.table("users").insert({
                    "username": uname,
                    "password_hash": pwd_hash,
                    "salt": salt,
                    "role": assigned_role,
                }).execute()
            except Exception:
                pass

    return True, "Registration successful.", {
        "id": user_id,
        "username": uname,
        "role": assigned_role,
    }


def authenticate_user(
    username: str,
    password: str,
    db_path: Path | str = DB_FILE,
) -> tuple[bool, str, dict[str, Any] | None]:
    """Authenticate username and password against stored PBKDF2 hash (Supabase + SQLite)."""
    uname = username.strip()
    if not uname or not password:
        return False, "Username and password are required.", None

    # 1. Attempt Supabase authentication if available (only in production database mode)
    if _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                res = client.table("users").select("id, username, password_hash, salt, role").ilike("username", uname).execute()
                if res.data and len(res.data) > 0:
                    row = res.data[0]
                    if verify_password(password, row["password_hash"], row["salt"]):
                        return True, "Authentication successful (Supabase).", {
                            "id": row["id"],
                            "username": row["username"],
                            "role": row["role"],
                        }
                    return False, "Invalid username or password.", None
            except Exception:
                pass  # Fall back to local SQLite on network failure

    # 2. SQLite authentication
    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, username, password_hash, salt, role
                FROM users
                WHERE username = ?;
            """, (uname,))
            row = cursor.fetchone()

            if not row:
                return False, "Invalid username or password.", None

            user_id, db_uname, db_hash, salt, role = row
            if verify_password(password, db_hash, salt):
                return True, "Authentication successful.", {
                    "id": user_id,
                    "username": db_uname,
                    "role": role,
                }
            return False, "Invalid username or password.", None
    except Exception as e:
        return False, f"Authentication error: {e}", None


def get_user_by_id(user_id: int, db_path: Path | str = DB_FILE) -> dict[str, Any] | None:
    """Retrieve user dictionary by ID."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, role, created_at FROM users WHERE id = ?;", (user_id,))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None


def list_all_users(db_path: Path | str = DB_FILE) -> list[dict[str, Any]]:
    """List all user accounts with sub-millisecond local speed, falling back to Supabase if local is empty."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT u.id, u.username, u.role, u.created_at, COUNT(p.id) as prediction_count
            FROM users u
            LEFT JOIN predictions p ON u.id = p.user_id
            GROUP BY u.id
            ORDER BY u.id ASC;
        """)
        local_users = [dict(row) for row in cursor.fetchall()]

    if local_users:
        return local_users

    if _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                res = client.table("users").select("id, username, role, created_at").order("id", desc=False).execute()
                if res.data:
                    users_list = res.data
                    try:
                        preds_res = client.table("predictions").select("user_id").execute()
                        counts: dict[int, int] = {}
                        for p in preds_res.data or []:
                            uid = p.get("user_id")
                            counts[uid] = counts.get(uid, 0) + 1
                        for u in users_list:
                            u["prediction_count"] = counts.get(u["id"], 0)
                    except Exception:
                        for u in users_list:
                            u["prediction_count"] = 0
                    return users_list
            except Exception:
                pass

    return []


def save_prediction_record(
    user_id: int,
    image_filename: str,
    predicted_category: str,
    confidence_percent: float,
    db_path: Path | str = DB_FILE,
) -> int:
    """Save prediction with record ownership to SQLite database and Supabase."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf = round(confidence_percent, 2)

    # 1. Save to local SQLite
    rec_id = 0
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO predictions (user_id, timestamp, image_filename, predicted_category, confidence_percent)
            VALUES (?, ?, ?, ?, ?);
        """, (user_id, now_str, image_filename, predicted_category, conf))
        rec_id = cursor.lastrowid
        conn.commit()

    # 2. Save to Supabase (only in production database mode)
    if _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                client.table("predictions").insert({
                    "user_id": user_id,
                    "image_filename": image_filename,
                    "predicted_category": predicted_category,
                    "confidence_percent": conf,
                }).execute()
            except Exception:
                pass

    return rec_id


def get_predictions_for_user(
    user_id: int,
    is_admin: bool = False,
    db_path: Path | str = DB_FILE,
) -> list[dict[str, Any]]:
    """Fetch prediction records with sub-millisecond local speed, falling back to Supabase if local is empty."""
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        if is_admin:
            cursor.execute("""
                SELECT p.id, p.timestamp, p.image_filename, p.predicted_category, p.confidence_percent, u.username as owner
                FROM predictions p
                JOIN users u ON p.user_id = u.id
                ORDER BY p.id DESC;
            """)
        else:
            cursor.execute("""
                SELECT id, timestamp, image_filename, predicted_category, confidence_percent
                FROM predictions
                WHERE user_id = ?
                ORDER BY id DESC;
            """, (user_id,))
        local_records = [dict(row) for row in cursor.fetchall()]

    if local_records:
        return local_records

    # Remote Supabase fallback if local database has no records
    if _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                query = client.table("predictions").select("id, timestamp, image_filename, predicted_category, confidence_percent, user_id")
                if not is_admin:
                    query = query.eq("user_id", user_id)
                res = query.order("id", desc=True).execute()
                if res.data is not None and len(res.data) > 0:
                    return res.data
            except Exception:
                pass

    return []


def get_stats_for_user(
    user_id: int,
    is_admin: bool = False,
    db_path: Path | str = DB_FILE,
) -> dict[str, Any]:
    """Calculate aggregate statistics respecting ownership boundaries."""
    records = get_predictions_for_user(user_id=user_id, is_admin=is_admin, db_path=db_path)
    if not records:
        return {
            "total_analyses": 0,
            "distinct_classes": 0,
            "latest_date": "No records yet",
            "avg_confidence": 0.0,
        }

    total = len(records)
    categories = {r.get("predicted_category", "") for r in records if r.get("predicted_category")}
    confidences = [float(r.get("confidence_percent", 0.0)) for r in records]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    latest = records[0].get("timestamp", "Recent")

    return {
        "total_analyses": total,
        "distinct_classes": len(categories),
        "latest_date": latest,
        "avg_confidence": round(avg_conf, 1),
    }


def get_disease_records(db_path: Path | str = DB_FILE) -> list[dict[str, Any]]:
    """Fetch all disease knowledge entries from database with treatment and prevention data (sub-millisecond fast)."""
    # 1. Fast local SQLite read first
    with get_db_connection(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(diseases);")
        cols = {col[1] for col in cursor.fetchall()}

        if "treatment_json" in cols and "prevention_json" in cols:
            cursor.execute("SELECT id, disease_key, title, description, symptoms, care_json, severity, crop, pathogen, treatment_json, prevention_json FROM diseases ORDER BY title ASC;")
        else:
            cursor.execute("SELECT id, disease_key, title, description, symptoms, care_json, severity FROM diseases ORDER BY title ASC;")

        rows = cursor.fetchall()
        if len(rows) >= 38:
            result = []
            for r in rows:
                d = dict(r)
                try:
                    d["care"] = json.loads(d.get("care_json", "[]"))
                except Exception:
                    d["care"] = []

                try:
                    d["treatment"] = json.loads(d.get("treatment_json") or "[]")
                except Exception:
                    d["treatment"] = []
                if not d["treatment"] and d["care"]:
                    d["treatment"] = d["care"][:2] if len(d["care"]) >= 2 else d["care"]

                try:
                    d["prevention"] = json.loads(d.get("prevention_json") or "[]")
                except Exception:
                    d["prevention"] = []
                if not d["prevention"] and len(d["care"]) > 2:
                    d["prevention"] = d["care"][2:]

                result.append(d)
            return result

    # 2. Remote Supabase fallback if local database has insufficient records
    if _is_default_db(db_path):
        client = get_supabase_client()
        if client:
            try:
                res = client.table("diseases").select("*").order("title").execute()
                if res.data and len(res.data) >= 38:
                    result = []
                    for d in res.data:
                        c = d.get("care_json")
                        d["care"] = json.loads(c) if isinstance(c, str) else (c or [])
                        t = d.get("treatment_json")
                        d["treatment"] = json.loads(t) if isinstance(t, str) else (t or (d["care"][:2] if len(d["care"]) >= 2 else d["care"]))
                        p = d.get("prevention_json")
                        d["prevention"] = json.loads(p) if isinstance(p, str) else (p or (d["care"][2:] if len(d["care"]) > 2 else []))
                        result.append(d)
                    return result
            except Exception:
                pass

    # 3. In-memory DISEASE_INFO fallback
    from disease_info import DISEASE_INFO
    res = []
    for k, v in DISEASE_INFO.items():
        entry = v.copy()
        entry["disease_key"] = k
        res.append(entry)
    return res


def save_disease_record(
    disease_key: str,
    title: str,
    description: str,
    symptoms: str,
    care_list: list[str],
    severity: str = "Moderate",
    crop: str = "",
    pathogen: str = "",
    treatment_list: list[str] | None = None,
    prevention_list: list[str] | None = None,
    db_path: Path | str = DB_FILE,
) -> bool:
    """Insert or update a disease knowledge entry in SQLite and Supabase."""
    if treatment_list is None:
        treatment_list = care_list[:2] if len(care_list) >= 2 else care_list
    if prevention_list is None:
        prevention_list = care_list[2:] if len(care_list) > 2 else []

    care_str = json.dumps(care_list)
    treat_str = json.dumps(treatment_list)
    prev_str = json.dumps(prevention_list)

    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(diseases);")
            cols = {col[1] for col in cursor.fetchall()}
            if "treatment_json" in cols:
                cursor.execute("""
                    INSERT INTO diseases (disease_key, title, description, symptoms, care_json, severity, crop, pathogen, treatment_json, prevention_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(disease_key) DO UPDATE SET
                        title=excluded.title,
                        description=excluded.description,
                        symptoms=excluded.symptoms,
                        care_json=excluded.care_json,
                        severity=excluded.severity,
                        crop=excluded.crop,
                        pathogen=excluded.pathogen,
                        treatment_json=excluded.treatment_json,
                        prevention_json=excluded.prevention_json;
                """, (disease_key, title, description, symptoms, care_str, severity, crop, pathogen, treat_str, prev_str))
            else:
                cursor.execute("""
                    INSERT INTO diseases (disease_key, title, description, symptoms, care_json, severity)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(disease_key) DO UPDATE SET
                        title=excluded.title,
                        description=excluded.description,
                        symptoms=excluded.symptoms,
                        care_json=excluded.care_json,
                        severity=excluded.severity;
                """, (disease_key, title, description, symptoms, care_str, severity))
            conn.commit()

        if _is_default_db(db_path):
            client = get_supabase_client()
            if client:
                try:
                    payload = {
                        "disease_key": disease_key,
                        "title": title,
                        "description": description,
                        "symptoms": symptoms,
                        "care_json": care_list,
                        "severity": severity,
                    }
                    client.table("diseases").upsert(payload).execute()
                except Exception:
                    pass
        return True
    except Exception:
        return False


def sync_local_to_supabase(db_path: Path | str = DB_FILE) -> dict[str, Any]:
    """Safely synchronize local SQLite records into Supabase PostgreSQL Cloud database."""
    client = get_supabase_client()
    if not client:
        return {
            "success": False,
            "error": "Supabase is not configured or not connected. Check SUPABASE_URL and SUPABASE_KEY.",
            "synced": {"users": 0, "diseases": 0, "predictions": 0},
        }

    synced_users = 0
    synced_diseases = 0
    synced_predictions = 0

    try:
        with get_db_connection(db_path) as conn:
            cursor = conn.cursor()

            # 1. Sync Users
            cursor.execute("SELECT username, password_hash, salt, role, created_at FROM users;")
            for u in cursor.fetchall():
                try:
                    client.table("users").upsert({
                        "username": u["username"],
                        "password_hash": u["password_hash"],
                        "salt": u["salt"],
                        "role": u["role"],
                        "created_at": u["created_at"],
                    }, on_conflict="username").execute()
                    synced_users += 1
                except Exception:
                    pass

            # 2. Sync Diseases
            cursor.execute("SELECT disease_key, title, description, symptoms, care_json, severity FROM diseases;")
            for d in cursor.fetchall():
                try:
                    care_val = json.loads(d["care_json"]) if isinstance(d["care_json"], str) else d["care_json"]
                    client.table("diseases").upsert({
                        "disease_key": d["disease_key"],
                        "title": d["title"],
                        "description": d["description"],
                        "symptoms": d["symptoms"],
                        "care_json": care_val,
                        "severity": d["severity"],
                    }, on_conflict="disease_key").execute()
                    synced_diseases += 1
                except Exception:
                    pass

            # 3. Sync Predictions
            cursor.execute("SELECT user_id, timestamp, image_filename, predicted_category, confidence_percent FROM predictions;")
            for p in cursor.fetchall():
                try:
                    client.table("predictions").insert({
                        "user_id": p["user_id"],
                        "timestamp": p["timestamp"],
                        "image_filename": p["image_filename"],
                        "predicted_category": p["predicted_category"],
                        "confidence_percent": p["confidence_percent"],
                    }).execute()
                    synced_predictions += 1
                except Exception:
                    pass

        return {
            "success": True,
            "error": None,
            "synced": {
                "users": synced_users,
                "diseases": synced_diseases,
                "predictions": synced_predictions,
            },
        }
    except Exception as err:
        return {
            "success": False,
            "error": str(err),
            "synced": {
                "users": synced_users,
                "diseases": synced_diseases,
                "predictions": synced_predictions,
            },
        }


def store_private_upload(
    file_bytes: bytes,
    filename: str,
    user_id: int,
    content_type: str = "image/jpeg",
) -> tuple[bool, str, str]:
    """Store private uploaded image or report in Supabase storage or isolated local folder."""
    from security_utils import sanitize_filename

    safe_name = sanitize_filename(filename)
    storage_path = f"user_{user_id}/{safe_name}"

    # 1. Attempt Supabase Storage
    client = get_supabase_client()
    if client:
        try:
            client.storage.from_("plantcare-uploads").upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": content_type, "upsert": "true"},
            )
            return True, storage_path, "cloud"
        except Exception:
            pass

    # 2. Local fallback in isolated user directory
    local_dir = Path(__file__).parent / "uploads" / f"user_{user_id}"
    local_dir.mkdir(parents=True, exist_ok=True)
    local_file = local_dir / safe_name
    local_file.write_bytes(file_bytes)
    return True, str(local_file), "local"
