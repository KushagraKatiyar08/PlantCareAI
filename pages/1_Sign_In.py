"""PlantCare AI — Dedicated User Sign-In & Registration Portal.

A dedicated standalone authentication site directly connected to the Supabase PostgreSQL
Cloud Database with local SQLite fallback, PBKDF2-HMAC-SHA256 password hashing,
and seamless session sharing with the main PlantCare AI dashboard.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

# Ensure repository root is on Python module search path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from auth_db import (
    authenticate_user,
    get_database_status,
    get_supabase_client,
    init_db,
    register_user,
)

# Initialize database schema
init_db()

# Page configuration for standalone sign-in site
st.set_page_config(
    page_title="Sign In — PlantCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Inject design system stylesheet
css_path = REPO_ROOT / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def render_dashboard_link(label: str, icon: str = "✦", is_primary: bool = False):
    """Render link back to main dashboard with graceful fallback."""
    try:
        st.page_link("app.py", label=label, icon=icon)
    except Exception:
        bg_style = "background: #ffffff; color: #0a0a0a;" if is_primary else "background: transparent; color: #ffffff; border: 1px solid rgba(255,255,255,0.25);"
        st.markdown(
            f"""
            <a href="/" target="_self" style="display: inline-flex; align-items: center; gap: 8px; padding: 7px 18px; border-radius: 9999px; font-weight: 400; font-size: 13.5px; text-decoration: none; {bg_style}">
              <span>{icon}</span> <span>{label}</span>
            </a>
            """,
            unsafe_allow_html=True,
        )


# Top Bar / Navigation Back to Main App
nav_c1, nav_c2 = st.columns([1.5, 1], vertical_alignment="center")
with nav_c1:
    st.markdown(
        """
        <div class="pc-brand-group">
          <div class="pc-brand-logo">✦</div>
          <div style="display: flex; flex-direction: column;">
            <span class="pc-brand-title">PLANTCARE AI</span>
            <span class="pc-brand-descriptor">GROWER AUTHENTICATION PORTAL</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with nav_c2:
    st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
    render_dashboard_link("← Return to Dashboard", icon="✦")
    st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# Query Supabase connection status
db_status = get_database_status()

# Hero Banner
st.markdown(
    """
    <div class="pc-hero-container" style="padding: 32px 36px; margin-bottom: 24px;">
      <div class="pc-eyebrow">IDENTITY SERVICE // CLOUD AUTHENTICATION</div>
      <div class="pc-hero-headline" style="font-size: 34px; margin-bottom: 8px;">Grower and Administrator Access</div>
      <div class="pc-hero-subtext" style="font-size: 15px; margin-bottom: 0;">
        Access your private crop diagnosis history, sync telemetry with Supabase Cloud, and secure your plant health records.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Centered Sign In Layout
center_col1, center_col2, center_col3 = st.columns([1, 1.8, 1])

with center_col2:
    # Live Supabase Connection Badge
    if db_status["supabase_connected"]:
        st.markdown(
            f"""
            <div class="pc-status-pill pc-status-supabase pc-animate-fade" style="width: 100%; justify-content: center; margin-bottom: 18px;">
              <span class="pc-status-dot"></span>
              <strong>Live: Connected to Supabase Cloud Database</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="pc-status-pill pc-status-sqlite pc-animate-fade" style="width: 100%; justify-content: center; margin-bottom: 18px;">
              <span class="pc-status-dot"></span>
              <strong>Active: Local SQLite Engine (Supabase Cloud Configurable)</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Auth Container Card
    with st.container(border=True):
        # If already authenticated, show active profile status
        active_user = st.session_state.get("user")
        if active_user:
            u_name = active_user.get("username", "User")
            u_initial = u_name[0].upper() if u_name else "U"
            u_role = active_user.get("role", "user")

            st.markdown(
                f"""
                <div style="text-align: center; padding: 12px 0 20px 0;">
                  <div class="pc-user-avatar-lg">
                    {u_initial}
                    <span class="pc-avatar-badge-lg">✓</span>
                  </div>
                  <div class="pc-auth-title">Currently Signed In as {u_name}</div>
                  <div class="pc-auth-subtitle">Role: <strong>{u_role.title()}</strong> • ID: #{active_user.get('id')}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            b_c1, b_c2 = st.columns(2)
            with b_c1:
                render_dashboard_link("🌿 Go to Diagnostic Dashboard", icon="🌱", is_primary=True)
            with b_c2:
                if st.button("Sign Out", type="secondary", use_container_width=True, key="btn_page_signout"):
                    st.session_state.user = None
                    st.rerun()

        else:
            # Dual-tab Sign In / Registration
            tab_signin, tab_signup, tab_supabase = st.tabs(["Sign In", "Create Account", "☁️ Supabase Setup"])

            with tab_signin:
                st.markdown("##### Enter Credentials")
                si_user = st.text_input("Username", key="standalone_si_user", placeholder="Enter username")
                si_pass = st.text_input("Password", type="password", key="standalone_si_pass", placeholder="Enter password")
                st.write("")

                if st.button("Sign In to Account", type="primary", key="btn_standalone_si_submit", use_container_width=True):
                    if not (si_user and si_pass):
                        st.error("Please provide both username and password.")
                    else:
                        ok, msg, user_obj = authenticate_user(si_user, si_pass)
                        if ok and user_obj:
                            st.session_state.user = user_obj
                            st.session_state.pending_nav_page = "Diagnose"
                            st.success(f"Welcome back, {user_obj['username']}! Redirecting...")
                            render_dashboard_link("👉 Open Dashboard with Active Session", icon="🌿", is_primary=True)
                            st.rerun()
                        else:
                            st.error(msg)

        with tab_signup:
            st.markdown("##### Register New Account in Supabase")
            st.caption("Create a private grower account to persist your leaf diagnostics and sync across devices.")
            su_user = st.text_input("Desired Username", key="standalone_su_user", placeholder="Alphanumeric (3-30 chars)")
            su_pass = st.text_input("Choose Password", type="password", key="standalone_su_pass", placeholder="Min 6 characters")

            if st.button("Create Account in Database", type="primary", key="btn_standalone_su_submit", use_container_width=True):
                if not (su_user and su_pass):
                    st.error("Please enter a username and password.")
                else:
                    ok, msg, user_obj = register_user(su_user, su_pass, role="user", is_admin_request=False)
                    if ok and user_obj:
                        st.session_state.user = user_obj
                        st.session_state.pending_nav_page = "Diagnose"
                        st.success(f"Account for '{su_user}' successfully created in database!")
                        st.rerun()
                    else:
                        st.error(msg)

        with tab_supabase:
            st.markdown("##### Supabase Database Connection")
            st.caption("Configure or verify your Supabase project parameters:")

            env_u = os.environ.get("SUPABASE_URL", "")
            env_k = os.environ.get("SUPABASE_KEY") or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "")

            cfg_url = st.text_input("Supabase Project URL", value=env_u, placeholder="https://your-project.supabase.co", key="standalone_cfg_url")
            cfg_key = st.text_input("Supabase API Key", value=env_k, type="password", placeholder="eyJhbGciOiJIUzI1NiIsIn...", key="standalone_cfg_key")

            if st.button("💾 Connect Supabase Database", type="primary", key="btn_standalone_cfg_save", use_container_width=True):
                if not (cfg_url.strip() and cfg_key.strip()):
                    st.error("Please provide both Supabase URL and API Key.")
                else:
                    clean_u = cfg_url.strip().rstrip("/")
                    clean_k = cfg_key.strip()
                    connected = False
                    conn_err = None

                    # 1. Probe via official supabase-py client
                    try:
                        from supabase import create_client
                        test_client = create_client(clean_u, clean_k)
                        test_client.table("users").select("id").limit(1).execute()
                        connected = True
                    except ImportError:
                        pass
                    except Exception as err:
                        conn_err = str(err)

                    # 2. Fallback probe via direct HTTPS REST endpoint
                    if not connected:
                        try:
                            import urllib.request
                            probe_url = f"{clean_u}/rest/v1/users?select=id&limit=1"
                            req = urllib.request.Request(
                                probe_url,
                                headers={
                                    "apikey": clean_k,
                                    "Authorization": f"Bearer {clean_k}",
                                    "Accept": "application/json",
                                },
                            )
                            with urllib.request.urlopen(req, timeout=10) as resp:
                                if resp.status in (200, 206):
                                    connected = True
                        except Exception as rest_err:
                            if not conn_err:
                                conn_err = str(rest_err)

                    if connected:
                        env_file = REPO_ROOT / ".env"
                        env_file.write_text(f'SUPABASE_URL="{clean_u}"\nSUPABASE_KEY="{clean_k}"\n', encoding="utf-8")
                        os.environ["SUPABASE_URL"] = clean_u
                        os.environ["SUPABASE_KEY"] = clean_k

                        st.success("Connected to Supabase Cloud Database! Credentials saved.")
                        st.rerun()
                    else:
                        st.error(f"Failed to connect to Supabase: {conn_err or 'Check URL and Key'}. Ensure you have executed supabase_schema.sql in your Supabase SQL Editor.")

# Site Footer
st.markdown(
    """
    <footer class="pc-footer" style="margin-top: 56px;">
      <div class="pc-footer-inner">
        <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
          <span class="pc-eyebrow" style="color: #8e9192;">© 2025 PLANTCARE AI // EMPIRICAL PLANT PATHOLOGY AT EDGE SCALE</span>
          <span style="color: #212327;">/</span>
          <span class="pc-pill" style="font-size: 10px; color: #9ca3af;">IDENTITY CLOUD SERVICE</span>
        </div>
        <div style="display: flex; align-items: center; gap: 20px;" class="pc-eyebrow">
          <a href="/" target="_self" style="color: #ffffff; text-decoration: none;">← Return to Main Dashboard</a>
        </div>
      </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
