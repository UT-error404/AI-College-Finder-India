#!/usr/bin/env python3
"""
AI College Finder — India Edition (Professional Redesign)
Powered by Google Gemini AI + Live Google Search

Run locally:
    pip install streamlit google-genai pandas
    streamlit run College_finder.py

Get your free Gemini API key at: https://aistudio.google.com/apikey
"""

import os
import sys
import re
import json
import time
import importlib
import subprocess

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI College Finder — India",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ════════════════════════════════════════════════════════════════════════════
#  SDK BOOTSTRAP
# ════════════════════════════════════════════════════════════════════════════

def _import_genai():
    try:
        from google import genai as _genai
        from google.genai import types as _types
        return _genai, _types
    except ImportError:
        pass
    with st.spinner("Setting up the Gemini SDK (one-time setup)…"):
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "google-genai"]
            )
            importlib.invalidate_caches()
            from google import genai as _genai
            from google.genai import types as _types
            return _genai, _types
        except Exception as exc:
            st.error(
                "**Could not install `google-genai` automatically.**\n\n"
                f"Error: `{exc}`\n\n"
                "Fix: run `pip install -r requirements.txt` then restart."
            )
            st.stop()


genai, types = _import_genai()


# ════════════════════════════════════════════════════════════════════════════
#  BACKEND API KEY
# ════════════════════════════════════════════════════════════════════════════
#  The Gemini API key lives on the server/deployment side only — end users
#  never see or enter it. Set it using ONE of the following (checked in
#  this order):
#
#   1) Streamlit secrets file:  .streamlit/secrets.toml
#         GEMINI_API_KEY = "your-key-here"
#
#   2) Environment variable (e.g. set on your hosting platform):
#         export GEMINI_API_KEY="your-key-here"
#
#  Just put your real key in one of those two places — nothing else in
#  this file needs to change.

def _get_backend_api_key() -> str:
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")


GEMINI_API_KEY = _get_backend_api_key()


# ════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

APP_VERSION  = "3.0.0"
MODEL_OPTIONS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-3-flash-preview",
]
DEFAULT_MODEL = "gemini-2.5-flash"

INTEREST_AREAS = {
    "1":  "Engineering (B.Tech / B.E.)",
    "2":  "Computer Science & AI / Data Science",
    "3":  "Medicine & MBBS / BDS / BAMS",
    "4":  "Finance, Commerce & Economics (B.Com / BBA)",
    "5":  "Management & MBA",
    "6":  "Design, Fine Arts & Architecture",
    "7":  "Law (BA LLB / BBA LLB / LLB)",
    "8":  "Science (B.Sc — Physics / Chemistry / Biology / Maths)",
    "9":  "Humanities, Social Sciences & Liberal Arts",
    "10": "Agriculture, Forestry & Environment",
    "11": "Pharmacy (B.Pharm / Pharm.D)",
    "12": "Journalism, Media & Mass Communication",
    "13": "Hotel Management & Hospitality",
    "14": "Education & Teaching (B.Ed / D.El.Ed)",
    "15": "Aviation & Aeronautics",
}

LOCATIONS = {
    "1":  "Delhi NCR (Delhi, Noida, Gurgaon, Faridabad)",
    "2":  "Mumbai (Mumbai, Navi Mumbai, Pune)",
    "3":  "Bangalore (Bengaluru, Mysore)",
    "4":  "Chennai (Chennai, Coimbatore, Vellore)",
    "5":  "Hyderabad (Hyderabad, Secunderabad, Warangal)",
    "6":  "Kolkata (Kolkata, Kharagpur, Durgapur)",
    "7":  "Ahmedabad (Ahmedabad, Surat, Vadodara)",
    "8":  "Jaipur (Jaipur, Jodhpur, Udaipur)",
    "9":  "Chandigarh (Chandigarh, Mohali, Panchkula)",
    "10": "Lucknow (Lucknow, Kanpur, Allahabad)",
    "11": "Bhopal (Bhopal, Indore, Jabalpur)",
    "12": "Northeast India (Guwahati, Shillong, Imphal)",
    "13": "South India — Any city",
    "14": "North India — Any city",
    "15": "Anywhere in India (No location preference)",
}

ENTRANCE_EXAMS = {
    "1":  "JEE Main",
    "2":  "JEE Advanced",
    "3":  "NEET UG",
    "4":  "CAT",
    "5":  "CLAT",
    "6":  "NATA / JEE Paper 2 (Architecture)",
    "7":  "CUET UG",
    "8":  "GATE",
    "9":  "GMAT / GRE",
    "10": "MAT / XAT / SNAP",
    "11": "State CET (e.g., MH-CET, KCET, AP EAMCET)",
    "12": "No entrance exam / Direct admission",
    "13": "Not yet decided",
}

COURSE_TYPES = {
    "1": "Undergraduate (UG) — 3 or 4 years",
    "2": "Postgraduate (PG) — 2 years (M.Tech / MBA / M.Sc / MA)",
    "3": "Integrated (5-year dual degree — B.Tech+M.Tech / BA+LLB)",
    "4": "Diploma / Certificate",
    "5": "PhD / Research",
}

BUDGET_RANGES = {
    "1":  ("Under ₹1 Lakh/year",       0,         100_000),
    "2":  ("₹1 – 3 Lakh/year",         100_000,   300_000),
    "3":  ("₹3 – 5 Lakh/year",         300_000,   500_000),
    "4":  ("₹5 – 8 Lakh/year",         500_000,   800_000),
    "5":  ("₹8 – 12 Lakh/year",        800_000,   1_200_000),
    "6":  ("₹12 – 20 Lakh/year",       1_200_000, 2_000_000),
    "7":  ("₹20 – 30 Lakh/year",       2_000_000, 3_000_000),
    "8":  ("Above ₹30 Lakh/year",      3_000_000, 9_999_999),
}

SCHOLARSHIP_OPTIONS = {
    "1": "I need a full scholarship / fee waiver",
    "2": "I qualify for a partial scholarship (merit / category)",
    "3": "I plan to take an education loan",
    "4": "I can self-fund without any financial aid",
    "5": "I'm unsure — I'd like to explore options",
}

PLACEMENT_SECTORS = {
    "1":  "Technology / IT / Software",
    "2":  "Finance / Banking / BFSI",
    "3":  "Consulting / Management",
    "4":  "Healthcare / Pharma / Biotech",
    "5":  "Government / Civil Services / PSU",
    "6":  "Design / Creative / Media",
    "7":  "Research / Academia",
    "8":  "Entrepreneurship / Startups",
    "9":  "Manufacturing / Core Engineering",
    "10": "No preference",
}

COLLEGE_TYPES = {
    "1": "Government / Public (IIT, NIT, AIIMS, Central University)",
    "2": "Private (Top private universities / autonomous colleges)",
    "3": "Deemed University",
    "4": "Autonomous College under State University",
    "5": "Any type — show me the best options",
}

HOSTEL_OPTIONS = {
    "1": "Yes — on-campus hostel preferred",
    "2": "No — I'll find accommodation myself",
    "3": "No preference",
}

EXTRACURRICULAR_OPTIONS = {
    "1": "Sports & athletics facilities",
    "2": "Research & innovation labs",
    "3": "Entrepreneurship / incubation cell",
    "4": "Cultural & arts activities",
    "5": "Industry internship tie-ups",
    "6": "International exchange programs",
    "7": "Strong alumni network",
    "8": "No specific preference",
}

BADGE_STYLES = {
    "Top Pick":    ("badge-top",   "#b91c1c", "#fef2f2"),
    "Best ROI":    ("badge-roi",   "#92400e", "#fffbeb"),
    "Great Value": ("badge-value", "#065f46", "#ecfdf5"),
    "Good Fit":    ("badge-good",  "#00574f", "#e6f5f4"),
    "Reach School":("badge-reach", "#5b21b6", "#f5f3ff"),
}


# ════════════════════════════════════════════════════════════════════════════
#  PROFESSIONAL CSS + ICON LIBRARY INJECTION
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<link rel="stylesheet"
      href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap"
      rel="stylesheet">

<style>
/* ── Base typography ────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide default Streamlit chrome ─────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.6rem !important; padding-bottom: 2rem !important; }

/* ── Force light color scheme app-wide (overrides OS/browser dark mode) ── */
:root, html { color-scheme: light !important; }

[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.stApp,
html,
body {
    background: #1c1c1e !important;
}
[data-testid="stHeader"] { background: transparent !important; }

/* ── App background ─────────────────────────────────────────── */
.main,
.main .block-container {
    background: #1c1c1e !important;
}

/* Force every native input/textarea/select inside the main content
   area to a dark surface, regardless of system light-mode defaults */
.main input,
.main textarea,
.main select {
    background: #2a2a2d !important;
    color: #e5e7eb !important;
    border: 1px solid #3f3f42 !important;
}

/* Streamlit's selectbox/multiselect are BaseWeb components, not native
   <select> tags — style their visible shell explicitly */
.main [data-baseweb="select"] > div,
.main [data-baseweb="base-input"] {
    background: #2a2a2d !important;
    color: #e5e7eb !important;
    border-color: #3f3f42 !important;
}
.main [data-baseweb="select"] span { color: #e5e7eb !important; }

/* Number input +/- step buttons */
.main [data-testid="stNumberInput"] button {
    background: #3f3f42 !important;
    color: #e5e7eb !important;
    border-color: #4b4b4f !important;
}

/* Dropdown / multiselect popover menus render in a portal at body level */
[data-baseweb="popover"] [data-baseweb="menu"],
[data-baseweb="menu"] {
    background: #2a2a2d !important;
}
[data-baseweb="menu"] li,
[role="option"] {
    background: #2a2a2d !important;
    color: #e5e7eb !important;
}
[role="option"]:hover,
[aria-selected="true"] {
    background: #00453f !important;
}

/* Main-content labels and body copy stay light on the new dark surface */
.main label,
.main label p,
.main label span,
.main label div,
.main [data-testid="stMarkdownContainer"] p,
.main [data-testid="stWidgetLabel"],
.main [data-testid="stWidgetLabel"] *,
.main [data-testid="stWidgetLabel"] p,
.main [data-testid="stWidgetLabel"] span,
.main [data-testid="stWidgetLabel"] div,
.main [data-testid="stWidgetLabel"] label,
.main [data-testid="stCaptionContainer"],
.main [data-testid="stCaptionContainer"] * {
    color: #e5e7eb !important;
}

/* Help-icon tooltip text on widget labels */
.main [data-testid="stWidgetLabel"] [data-testid="stTooltipIcon"] {
    color: #a1a1aa !important;
}

/* ── Sidebar ────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #00342f !important;
    border-right: 1px solid #00564e !important;
}
[data-testid="stSidebar"] * { color: #cdeae6 !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select,
[data-testid="stSidebar"] textarea {
    background: #00564e !important;
    border: 1px solid #007a6e !important;
    color: #f1f5f9 !important;
}
/* Sidebar's BaseWeb selectbox shell (e.g. Model Selection) — keep on-brand
   dark teal instead of falling back to Streamlit's default dark theme */
[data-testid="stSidebar"] [data-baseweb="select"] > div,
[data-testid="stSidebar"] [data-baseweb="base-input"] {
    background: #00564e !important;
    border-color: #007a6e !important;
    color: #f1f5f9 !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] span { color: #f1f5f9 !important; }

/* ── Sidebar brand block ──────────────────────────────────────*/
.sidebar-brand {
    font-size: 18px;
    font-weight: 700;
    color: #f1f5f9 !important;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 0.5rem 0 0.2rem;
}
.sidebar-brand i { color: #ff8a3d !important; font-size: 22px; }
.sidebar-tagline {
    font-size: 11px;
    color: #7fb8b0 !important;
    margin-bottom: 12px;
    padding-left: 2px;
}
.sidebar-section-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: #ff8a3d !important;
    padding: 14px 0 6px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.sidebar-section-label i { font-size: 12px; color: #ff8a3d !important; }
.sidebar-note {
    font-size: 11px;
    color: #6fa39b !important;
    line-height: 1.6;
    padding: 6px 0;
}

/* ── App header ─────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #00342f 0%, #00786f 55%, #00968c 100%);
    border-radius: 14px;
    padding: 2.2rem 2.4rem 1.8rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
}
.app-header::after {
    content: '';
    position: absolute;
    right: -40px; top: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(244,121,32,0.18) 0%, transparent 70%);
    border-radius: 50%;
}
.app-header-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #ffb066;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 6px;
}
.app-header h1 {
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    margin: 0 0 8px;
    line-height: 1.2;
    letter-spacing: -0.5px;
}
.app-header h1 span { color: #ffb066; }
.app-header p {
    font-size: 14px;
    color: rgba(248,250,252,0.66);
    margin: 0;
    max-width: 600px;
    line-height: 1.7;
}

/* ── Expander (college result cards + sources) ─────────────────
   Native Streamlit component — must be styled explicitly or it
   falls back to the ambient theme, which can render the header
   text the same color as its background (unreadable). ──────── */
[data-testid="stExpander"] {
    background: #2a2a2d !important;
    border: 1px solid #3f3f42 !important;
    border-radius: 12px !important;
    margin-bottom: 14px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    background: #232326 !important;
    color: #e5e7eb !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
}
[data-testid="stExpander"] summary:hover {
    background: #00453f !important;
    color: #eafff9 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary div {
    color: inherit !important;
}
[data-testid="stExpander"] svg { fill: #2dd4bf !important; }
[data-testid="stExpanderDetails"],
[data-testid="stExpander"] [data-testid="stExpanderDetails"] {
    background: #2a2a2d !important;
    color: #e5e7eb !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] p {
    color: #e5e7eb !important;
}

/* ── Section header ─────────────────────────────────────────── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 14px;
    font-weight: 600;
    color: #e5e7eb;
    padding: 10px 0 8px;
    border-bottom: 1.5px solid #3f3f42;
    margin: 20px 0 14px;
}
.section-header i {
    color: #2dd4bf;
    font-size: 16px;
}

/* ── Form card container ────────────────────────────────────── */
.form-card {
    background: #2a2a2d;
    border: 1px solid #3f3f42;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}

/* ── Primary CTA button ─────────────────────────────────────── */
div[data-testid="stButton"] > button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    letter-spacing: 0.3px !important;
    transition: all 0.22s ease !important;
    border-radius: 9px !important;
}
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #f47920 0%, #ff9a4d 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 13px 32px !important;
    box-shadow: 0 4px 14px rgba(244,121,32,0.38) !important;
}
div[data-testid="stButton"] > button[kind="primary"] p,
div[data-testid="stButton"] > button[kind="primary"] span,
div[data-testid="stButton"] > button[kind="primary"] div {
    color: #ffffff !important;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #d9650f 0%, #f47920 100%) !important;
    box-shadow: 0 6px 20px rgba(244,121,32,0.50) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(0) !important;
    box-shadow: 0 2px 8px rgba(244,121,32,0.35) !important;
}
/* Secondary / link buttons */
div[data-testid="stButton"] > button[kind="secondary"] {
    background: #2a2a2d !important;
    border: 1.5px solid #3f3f42 !important;
    color: #d4d4d8 !important;
}
div[data-testid="stButton"] > button[kind="secondary"] p,
div[data-testid="stButton"] > button[kind="secondary"] span,
div[data-testid="stButton"] > button[kind="secondary"] div {
    color: #d4d4d8 !important;
}
div[data-testid="stButton"] > button[kind="secondary"]:hover {
    border-color: #00786f !important;
    color: #2dd4bf !important;
    background: #00453f !important;
}

/* ── Link buttons (Official Website / Apply) ────────────────── */
[data-testid="stLinkButton"] > a {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: 8px !important;
    transition: all 0.18s !important;
    background: #2a2a2d !important;
    border: 1.5px solid #00968c !important;
    color: #2dd4bf !important;
}
[data-testid="stLinkButton"] > a:hover {
    background: #00786f !important;
    border-color: #00786f !important;
    color: #ffffff !important;
}

/* ── Download button ────────────────────────────────────────── */
div[data-testid="stDownloadButton"] > button {
    background: #0f2e1f !important;
    border: 1.5px solid #16a34a !important;
    color: #4ade80 !important;
    font-weight: 600 !important;
    border-radius: 9px !important;
}
div[data-testid="stDownloadButton"] > button:hover {
    background: #14532d !important;
}

/* ── Results section header ─────────────────────────────────── */
.results-banner {
    background: #2a2a2d;
    border: 1px solid #3f3f42;
    border-left: 4px solid #00968c;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
}
.results-banner-title {
    font-size: 17px;
    font-weight: 700;
    color: #e5e7eb;
    display: flex;
    align-items: center;
    gap: 10px;
}
.results-banner-title i { color: #2dd4bf; font-size: 20px; }
.results-count-pill {
    background: #00453f;
    border: 1px solid #00786f;
    color: #eafff9;
    font-size: 12px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
}

/* ── Comparison table ───────────────────────────────────────── */
.table-wrapper {
    overflow-x: auto;
    border-radius: 12px;
    border: 1px solid #3f3f42;
    box-shadow: 0 1px 4px rgba(0,0,0,0.25);
    margin-bottom: 24px;
    background: #2a2a2d;
}
.cmp-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
}
.cmp-table thead tr {
    background: #00342f;
}
.cmp-table thead th {
    padding: 12px 16px;
    text-align: left;
    font-weight: 600;
    font-size: 11px;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    color: #ffffff;
    white-space: nowrap;
}
.cmp-table tbody tr { border-bottom: 1px solid #3f3f42; }
.cmp-table tbody tr:last-child { border-bottom: none; }
.cmp-table tbody tr:hover td { background: #232326; }
.cmp-table tbody td {
    padding: 13px 16px;
    color: #d4d4d8;
    vertical-align: middle;
}
.cmp-table tbody tr:nth-child(even) td { background: #232326; }
.cmp-table tbody tr:nth-child(even):hover td { background: #2a2a2d; }

/* Rank cell */
.rank-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px; height: 28px;
    background: #f47920;
    color: #fff;
    border-radius: 50%;
    font-size: 12px;
    font-weight: 700;
}

/* College name cell */
.col-name { font-weight: 600; color: #e5e7eb; font-size: 13px; }
.col-meta { font-size: 11px; color: #a1a1aa; margin-top: 2px; }

/* Fee cells */
.fee-ok   { color: #4ade80; font-weight: 600; }
.fee-over { color: #f87171; font-weight: 600; }
.fee-over::after { content: ' !'; }

/* Mini score bar */
.score-wrap { display: flex; align-items: center; gap: 7px; min-width: 100px; }
.mini-bar-track {
    flex: 1; height: 5px;
    background: #3f3f42;
    border-radius: 3px;
    overflow: hidden;
}
.mini-bar-fill { height: 100%; border-radius: 3px; }
.score-label { font-size: 11px; font-weight: 600; color: #d4d4d8; min-width: 30px; }

/* Fit badges */
.fit-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
}
.badge-top   { background: #3f1d1d; color: #fca5a5; border: 1px solid #7f1d1d; }
.badge-roi   { background: #3f2d0f; color: #fcd34d; border: 1px solid #78350f; }
.badge-value { background: #0f3f2a; color: #6ee7b7; border: 1px solid #065f46; }
.badge-good  { background: #00453f; color: #5eead4; border: 1px solid #00786f; }
.badge-reach { background: #2e1065; color: #c4b5fd; border: 1px solid #4c1d95; }

/* Budget pill */
.budget-ok   { background:#0f3f2a; color:#6ee7b7; border:1px solid #065f46;
               padding:2px 8px; border-radius:20px; font-size:11px; font-weight:500; }
.budget-over { background:#3f1d1d; color:#fca5a5; border:1px solid #7f1d1d;
               padding:2px 8px; border-radius:20px; font-size:11px; font-weight:500; }
.budget-slight { background:#3f2d0f; color:#fcd34d; border:1px solid #78350f;
                 padding:2px 8px; border-radius:20px; font-size:11px; font-weight:500; }

/* ── Detail profile cards ───────────────────────────────────── */
.profile-card {
    background: #2a2a2d;
    border: 1px solid #3f3f42;
    border-radius: 12px;
    padding: 22px 24px;
    margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.25);
}
.profile-card-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    margin-bottom: 16px;
    padding-bottom: 14px;
    border-bottom: 1px solid #3f3f42;
}
.profile-col-name {
    font-size: 16px;
    font-weight: 700;
    color: #e5e7eb;
}
.profile-col-meta {
    font-size: 12px;
    color: #a1a1aa;
    margin-top: 3px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.profile-col-meta i { font-size: 13px; }

/* Metric tiles inside detail cards */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 16px;
}
.metric-tile {
    background: #232326;
    border: 1px solid #3f3f42;
    border-radius: 9px;
    padding: 11px 14px;
}
.metric-tile-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #a1a1aa;
    margin-bottom: 4px;
}
.metric-tile-value {
    font-size: 17px;
    font-weight: 700;
    color: #e5e7eb;
}
.metric-tile-sub { font-size: 11px; color: #a1a1aa; margin-top: 1px; }

/* Score bars in detail cards */
.score-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
}
.score-row-label {
    font-size: 12px;
    color: #a1a1aa;
    width: 110px;
    flex-shrink: 0;
}
.score-bar-track {
    flex: 1; height: 7px;
    background: #3f3f42;
    border-radius: 4px;
    overflow: hidden;
}
.score-bar-fill { height: 100%; border-radius: 4px; }
.score-row-pct {
    font-size: 12px;
    font-weight: 600;
    color: #d4d4d8;
    min-width: 36px;
    text-align: right;
}

/* Tags */
.tag-row { display: flex; flex-wrap: wrap; gap: 6px; margin: 6px 0; }
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 500;
    background: #3f3f42;
    color: #d4d4d8;
    border: 1px solid #4b4b4f;
}
.tag-green { background: #0f3f2a; color: #6ee7b7; border-color: #065f46; }
.tag-red   { background: #3f1d1d; color: #fca5a5; border-color: #7f1d1d; }
.tag-blue  { background: #00453f; color: #5eead4; border-color: #00786f; }

/* Insight box */
.insight-box {
    background: #00342f;
    border: 1px solid #00786f;
    border-left: 4px solid #00968c;
    border-radius: 9px;
    padding: 13px 16px;
    font-size: 13px;
    color: #d1fae5;
    line-height: 1.65;
    margin: 14px 0;
}
.insight-box-label {
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    color: #2dd4bf;
    margin-bottom: 5px;
    display: flex;
    align-items: center;
    gap: 5px;
}

/* Detail sub-labels */
.detail-label {
    font-size: 11px;
    font-weight: 600;
    color: #a1a1aa;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 4px;
}
.detail-value { font-size: 13px; color: #e5e7eb; }

/* Sources expander */
.source-link {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 0;
    border-bottom: 1px solid #3f3f42;
    font-size: 13px;
    color: #7dd3fc;
    text-decoration: none;
}
.source-link i { font-size: 14px; color: #71717a; }
.source-link:hover { color: #bae6fd; }

/* Divider */
.hr-line {
    border: none;
    border-top: 1px solid #3f3f42;
    margin: 16px 0;
}

/* Section title for results */
.results-section-title {
    font-size: 15px;
    font-weight: 700;
    color: #e5e7eb;
    display: flex;
    align-items: center;
    gap: 9px;
    margin: 22px 0 14px;
    padding-bottom: 10px;
    border-bottom: 1.5px solid #3f3f42;
}
.results-section-title i { color: #2dd4bf; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════════════════

def format_inr(amount) -> str:
    if amount is None:
        return "N/A"
    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return str(amount)
    if amount >= 10_00_000:
        return f"₹{amount/10_00_000:.1f}L"
    elif amount >= 1_00_000:
        return f"₹{amount/1_00_000:.1f}L"
    else:
        return f"₹{amount/1000:.0f}K"


def section_header(icon: str, text: str) -> None:
    """Render a professional section header using a Bootstrap Icon class."""
    st.markdown(
        f'<div class="section-header"><i class="bi {icon}"></i>{text}</div>',
        unsafe_allow_html=True,
    )


def fit_badge_html(badge: str) -> str:
    cls_map = {
        "Top Pick":    "badge-top",
        "Best ROI":    "badge-roi",
        "Great Value": "badge-value",
        "Good Fit":    "badge-good",
        "Reach School":"badge-reach",
    }
    cls = cls_map.get(badge, "badge-good")
    return f'<span class="fit-badge {cls}">{badge}</span>'


def score_bar_html(label: str, pct: float, color: str) -> str:
    p = min(100, max(0, int(pct or 0)))
    return f"""
    <div class="score-row">
      <span class="score-row-label">{label}</span>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:{p}%;background:{color};"></div>
      </div>
      <span class="score-row-pct">{p}%</span>
    </div>"""


def budget_pill_html(budget_fit: str) -> str:
    cls = {
        "Within budget": "budget-ok",
        "Slightly over": "budget-slight",
        "Over budget":   "budget-over",
    }.get(budget_fit, "budget-ok")
    icon = {
        "Within budget": "bi-check-circle-fill",
        "Slightly over": "bi-exclamation-circle-fill",
        "Over budget":   "bi-x-circle-fill",
    }.get(budget_fit, "bi-check-circle-fill")
    return f'<span class="{cls}"><i class="bi {icon}"></i> {budget_fit}</span>'


# ════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDER  (unchanged)
# ════════════════════════════════════════════════════════════════════════════

def build_prompt(p: dict) -> str:
    location_str = " OR ".join(p["locations"])
    sector_str   = ", ".join(p["target_sectors"])
    extras_str   = ", ".join(p["extras"])

    if p["course_type_key"] in ("1", "3"):
        score_section = f"""
  • 12th score       : {p.get('class12_score', 'N/A')}%   (Board: {p.get('board', 'N/A')})
  • 10th score       : {p.get('class10_score', 'N/A')}%"""
    else:
        score_section = f"""
  • UG Degree        : {p.get('graduation_degree', 'N/A')}
  • UG Score         : {p.get('graduation_score', 'N/A')}%"""

    return f"""
You are an expert Indian college counsellor with real-time access to college websites, NIRF rankings, official admission portals, and placement data across India.

══════════════════════════════════════════════════════════════
STUDENT PROFILE
══════════════════════════════════════════════════════════════
Name              : {p['name']}
Age               : {p['age']}
Course Level      : {p['course_type']}
Interest / Field  : {p['interest']}
Specialisation    : {p.get('specialisation', 'None specified')}
{score_section}
Entrance Exam     : {p['entrance_exam']}  →  Score/Rank: {p.get('entrance_score', 'N/A')}
Annual Budget     : {p['budget_label']}  (₹{p['budget_min']:,} – ₹{p['budget_max']:,})
Financial Aid     : {p['scholarship']}
Location          : {location_str}
Hostel Needed     : {p['hostel']}
College Type      : {p['college_type']}
Target Sector     : {sector_str}
Min. Expected CTC : {p['min_package']}
Campus Extras     : {extras_str}
Additional Notes  : {p.get('notes', 'None')}

══════════════════════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════════════════════
Use Google Search to:
1. Find the top colleges across India matching this student profile.
2. Retrieve LIVE data for each college from official sources:
   - Official college/university websites (.ac.in / .edu.in)
   - NIRF Rankings 2024 (nirfindia.org)
   - NAAC ratings, fee structures for 2024-25, placement packages,
     admission cutoffs, courses, specialisations, scholarships, hostel.
3. Cross-check eligibility: is the student's score/rank competitive?
4. Compute ROI = (average_placement_package / annual_fee) ratio.

══════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════
Return a STRICT JSON array of exactly 6 college recommendations.
No markdown. No preamble. No explanation outside the JSON.

Schema for each element:
{{
  "rank": 1,
  "name": "Full official college name",
  "short_name": "Abbreviation e.g. IIT-B",
  "location": "City, State",
  "college_type": "Government / Private / Deemed",
  "naac_grade": "A++ / A+ / A / B++ or null",
  "nirf_rank": 123,
  "annual_fee": 500000,
  "total_course_fee": 2000000,
  "course_offered": "B.Tech Computer Science (4 years)",
  "available_specialisations": ["AI & ML", "Cybersecurity", "Data Science"],
  "admission_cutoff": "JEE Main >= 95 percentile / 12th >= 75%",
  "avg_package_lpa": 12.5,
  "highest_package_lpa": 45.0,
  "top_recruiters": ["Google", "Microsoft", "Amazon"],
  "roi_ratio": 2.5,
  "interest_match": 88,
  "budget_fit": "Within budget",
  "scholarship_available": true,
  "scholarship_details": "Merit scholarships up to 100% tuition waiver",
  "hostel_available": true,
  "hostel_fee_annual": 120000,
  "overall_fit_score": 91,
  "fit_badge": "Top Pick",
  "key_strengths": ["Strength 1", "Strength 2", "Strength 3"],
  "key_concerns": ["Concern 1"],
  "personalised_insight": "3-4 sentence insight for {p['name']} explaining why this college suits their profile.",
  "official_website": "https://www.college.ac.in",
  "apply_link": "https://apply.college.ac.in"
}}

Rules:
- budget_fit: exactly one of "Within budget", "Slightly over", "Over budget"
- fit_badge: exactly one of "Top Pick", "Best ROI", "Great Value", "Good Fit", "Reach School"
- Score realism: do NOT recommend IITs/AIIMS/NLUs if score is clearly below cutoff.
- Colleges MUST be in/near [{location_str}] unless 'Anywhere in India' selected.
- Flag "Over budget" if annual_fee > Rs {p['budget_max']:,}.
- Sort descending by overall_fit_score.
- Output STRICTLY VALID JSON only. Nothing else.
""".strip()


# ════════════════════════════════════════════════════════════════════════════
#  GEMINI API CALL  (unchanged logic)
# ════════════════════════════════════════════════════════════════════════════

def _repair_json(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def _is_transient_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(w in msg for w in ("503", "unavailable", "overloaded", "429", "rate", "deadline", "timeout"))


def call_gemini_with_search(api_key: str, model_name: str, prompt: str, max_retries: int = 3):
    client   = genai.Client(api_key=api_key)
    response = None
    last_exc = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                    max_output_tokens=16000,
                ),
            )
            break
        except Exception as exc:
            last_exc = exc
            if _is_transient_error(exc) and attempt < max_retries - 1:
                wait_s = 3 * (2 ** attempt)
                st.toast(f"Server busy — retrying in {wait_s}s (attempt {attempt + 2}/{max_retries})…")
                time.sleep(wait_s)
                continue
            raise

    if response is None:
        raise last_exc

    try:
        finish_reason = str(response.candidates[0].finish_reason)
    except Exception:
        finish_reason = ""
    if "MAX_TOKENS" in finish_reason:
        raise ValueError(
            "Response was cut off at the token limit. Try again or switch to a smaller model output."
        )

    text = (response.text or "").strip()
    if not text:
        raise ValueError("Gemini returned no text content. Please try again.")

    clean = text
    if clean.startswith("```"):
        clean = "\n".join(clean.split("\n")[1:])
    if clean.endswith("```"):
        clean = "\n".join(clean.split("\n")[:-1])

    start = clean.find("[")
    end   = clean.rfind("]") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON array found in the response. Please try again.")

    raw_json = clean[start:end]
    try:
        colleges = json.loads(raw_json)
    except json.JSONDecodeError:
        try:
            colleges = json.loads(_repair_json(raw_json))
        except json.JSONDecodeError as exc:
            snippet = raw_json[:400] + ("…" if len(raw_json) > 400 else "")
            raise ValueError(
                f"Could not parse the JSON response ({exc}).\n\nFirst 400 chars:\n\n{snippet}"
            )

    if not isinstance(colleges, list) or len(colleges) == 0:
        raise ValueError("Parsed result is empty or not a list.")

    sources = []
    try:
        gm = response.candidates[0].grounding_metadata
        if gm and gm.grounding_chunks:
            for chunk in gm.grounding_chunks:
                web = getattr(chunk, "web", None)
                if web is not None:
                    sources.append({
                        "title": getattr(web, "title", None) or getattr(web, "uri", "Source"),
                        "uri":   getattr(web, "uri", ""),
                    })
    except Exception:
        pass

    return colleges, sources


# ════════════════════════════════════════════════════════════════════════════
#  RESULTS RENDERING
# ════════════════════════════════════════════════════════════════════════════

def render_comparison_table(colleges: list, budget_max: int) -> str:
    """Build a professional HTML comparison table for all colleges."""
    rows_html = ""
    for c in colleges:
        rank      = c.get("rank", "?")
        name      = c.get("name", "N/A")
        short     = c.get("short_name", "")
        location  = c.get("location", "N/A")
        ctype     = c.get("college_type", "")
        naac      = c.get("naac_grade", "")
        nirf      = c.get("nirf_rank")
        annual_fee= float(c.get("annual_fee", 0) or 0)
        avg_pkg   = c.get("avg_package_lpa", 0) or 0
        roi       = c.get("roi_ratio", 0) or 0
        int_match = int(c.get("interest_match", 0) or 0)
        fit_score = int(c.get("overall_fit_score", 0) or 0)
        badge     = c.get("fit_badge", "Good Fit")
        budget_fit= c.get("budget_fit", "Within budget")

        over_budget = bool(budget_max) and annual_fee > budget_max * 1.05
        fee_class   = "fee-over" if over_budget else "fee-ok"

        col_display = f'<div class="col-name">{short or name}</div>'
        meta_parts  = []
        if ctype:
            meta_parts.append(ctype)
        if naac:
            meta_parts.append(f"NAAC {naac}")
        if nirf:
            meta_parts.append(f"NIRF #{nirf}")
        col_meta = f'<div class="col-meta">{" · ".join(meta_parts)}</div>' if meta_parts else ""

        # Interest match mini bar
        int_fill  = f'<div class="mini-bar-fill" style="width:{int_match}%;background:#00786f;"></div>'
        fit_fill  = f'<div class="mini-bar-fill" style="width:{fit_score}%;background:#059669;"></div>'
        int_cell  = f'''<div class="score-wrap">
            <div class="mini-bar-track">{int_fill}</div>
            <span class="score-label">{int_match}%</span>
        </div>'''
        fit_cell  = f'''<div class="score-wrap">
            <div class="mini-bar-track">{fit_fill}</div>
            <span class="score-label">{fit_score}%</span>
        </div>'''

        roi_str = f"{float(roi):.1f}x" if isinstance(roi, (int, float)) else "N/A"

        rows_html += f"""
        <tr>
          <td><span class="rank-num">{rank}</span></td>
          <td>{col_display}{col_meta}</td>
          <td>{location.split(',')[0] if location else '—'}</td>
          <td class="{fee_class}">{format_inr(annual_fee)}</td>
          <td style="font-weight:600;color:#0e2a47;">{avg_pkg} LPA</td>
          <td style="font-weight:600;color:#f47920;">{roi_str}</td>
          <td>{int_cell}</td>
          <td>{fit_cell}</td>
          <td>{fit_badge_html(badge)}</td>
          <td>{budget_pill_html(budget_fit)}</td>
        </tr>"""

    return f"""
    <div class="table-wrapper">
      <table class="cmp-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Institution</th>
            <th>City</th>
            <th>Annual Fee</th>
            <th>Avg Package</th>
            <th>ROI</th>
            <th>Interest Match</th>
            <th>Overall Fit</th>
            <th>Status</th>
            <th>Budget</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>"""


def display_college_detail(college: dict, budget_max: int) -> None:
    """Render one college's full details inside a professional expandable panel."""
    rank       = college.get("rank", "?")
    name       = college.get("name", "Unknown College")
    short      = college.get("short_name", "")
    location   = college.get("location", "")
    ctype      = college.get("college_type", "")
    naac       = college.get("naac_grade", "")
    nirf       = college.get("nirf_rank")
    badge      = college.get("fit_badge", "Good Fit")

    annual_fee = float(college.get("annual_fee",  0) or 0)
    total_fee  = float(college.get("total_course_fee", 0) or 0)
    course     = college.get("course_offered", "N/A")
    specs      = college.get("available_specialisations", []) or []
    cutoff     = college.get("admission_cutoff", "N/A")

    avg_pkg    = college.get("avg_package_lpa", 0) or 0
    high_pkg   = college.get("highest_package_lpa", 0) or 0
    roi        = college.get("roi_ratio", 0) or 0
    recruiters = college.get("top_recruiters", []) or []

    int_match  = int(college.get("interest_match", 0) or 0)
    overall    = int(college.get("overall_fit_score", 0) or 0)
    budget_fit = college.get("budget_fit", "Within budget")

    scholarship= college.get("scholarship_available", False)
    sch_detail = college.get("scholarship_details", "")
    hostel     = college.get("hostel_available", False)
    hostel_fee = float(college.get("hostel_fee_annual", 0) or 0)

    strengths  = college.get("key_strengths", []) or []
    concerns   = college.get("key_concerns",  []) or []
    insight    = college.get("personalised_insight", "")
    website    = college.get("official_website", "")
    apply_link = college.get("apply_link", "")

    over_budget= bool(budget_max) and annual_fee > budget_max * 1.05

    # Expander label (no emojis)
    exp_label = f"#{rank}  {name}"
    if short and short != name:
        exp_label += f"  —  {short}"

    with st.expander(exp_label, expanded=(rank == 1)):

        # ── Top meta row ─────────────────────────────────────
        meta_parts = []
        if location:
            meta_parts.append(f'<i class="bi bi-geo-alt-fill"></i> {location}')
        if ctype:
            meta_parts.append(f'<i class="bi bi-building"></i> {ctype}')
        if naac:
            meta_parts.append(f'<i class="bi bi-patch-check-fill"></i> NAAC {naac}')
        if nirf:
            meta_parts.append(f'<i class="bi bi-bar-chart-fill"></i> NIRF #{nirf}')

        meta_str = "&nbsp;&nbsp;·&nbsp;&nbsp;".join(meta_parts)
        st.markdown(f"""
        <div style="display:flex;align-items:center;justify-content:space-between;
                    margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid #f1f5f9;">
          <div style="font-size:12px;color:#64748b;">{meta_str}</div>
          {fit_badge_html(badge)}
        </div>""", unsafe_allow_html=True)

        # ── Key metrics grid ──────────────────────────────────
        fee_color   = "#b91c1c" if over_budget else "#059669"
        fee_sub     = "Over budget" if over_budget else "Per year"
        roi_str     = f"{float(roi):.1f}x" if isinstance(roi, (int, float)) else "N/A"

        st.markdown(f"""
        <div class="metric-grid">
          <div class="metric-tile">
            <div class="metric-tile-label"><i class="bi bi-wallet2"></i>&nbsp;Annual Fee</div>
            <div class="metric-tile-value" style="color:{fee_color};">{format_inr(annual_fee)}</div>
            <div class="metric-tile-sub">{fee_sub}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label"><i class="bi bi-cash-stack"></i>&nbsp;Total Course Fee</div>
            <div class="metric-tile-value">{format_inr(total_fee) if total_fee else 'N/A'}</div>
            <div class="metric-tile-sub">Full programme</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label"><i class="bi bi-graph-up-arrow"></i>&nbsp;Avg Package</div>
            <div class="metric-tile-value" style="color:#059669;">{avg_pkg} LPA</div>
            <div class="metric-tile-sub">Highest: {high_pkg} LPA</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label"><i class="bi bi-calculator"></i>&nbsp;ROI Ratio</div>
            <div class="metric-tile-value" style="color:#f47920;">{roi_str}</div>
            <div class="metric-tile-sub">Placement ÷ Fee</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label"><i class="bi bi-award-fill"></i>&nbsp;Scholarship</div>
            <div class="metric-tile-value" style="font-size:13px;">{'Available' if scholarship else 'Not listed'}</div>
            <div class="metric-tile-sub">{sch_detail[:40] + '…' if sch_detail and len(sch_detail) > 40 else (sch_detail or '—')}</div>
          </div>
          <div class="metric-tile">
            <div class="metric-tile-label"><i class="bi bi-house-fill"></i>&nbsp;Hostel</div>
            <div class="metric-tile-value" style="font-size:13px;">{'Available' if hostel else 'Not listed'}</div>
            <div class="metric-tile-sub">{format_inr(hostel_fee) + '/year' if hostel and hostel_fee else '—'}</div>
          </div>
        </div>""", unsafe_allow_html=True)

        # ── Course & cutoff ───────────────────────────────────
        col_a, col_b = st.columns(2, gap="medium")
        with col_a:
            st.markdown(f"""
            <div class="detail-label"><i class="bi bi-mortarboard-fill"></i>&nbsp;Course Offered</div>
            <div class="detail-value" style="font-weight:600;">{course}</div>""",
            unsafe_allow_html=True)
            if specs:
                tags = "".join(f'<span class="tag tag-blue">{s}</span>' for s in specs[:5])
                st.markdown(f"""
                <div class="detail-label" style="margin-top:10px;">
                  <i class="bi bi-diagram-3-fill"></i>&nbsp;Specialisations
                </div>
                <div class="tag-row">{tags}</div>""", unsafe_allow_html=True)

        with col_b:
            st.markdown(f"""
            <div class="detail-label"><i class="bi bi-clipboard-check-fill"></i>&nbsp;Admission Cutoff</div>
            <div class="detail-value">{cutoff}</div>""", unsafe_allow_html=True)
            if recruiters:
                rec_tags = "".join(f'<span class="tag">{r}</span>' for r in recruiters[:5])
                st.markdown(f"""
                <div class="detail-label" style="margin-top:10px;">
                  <i class="bi bi-briefcase-fill"></i>&nbsp;Top Recruiters
                </div>
                <div class="tag-row">{rec_tags}</div>""", unsafe_allow_html=True)

        st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

        # ── Score bars ────────────────────────────────────────
        st.markdown(
            score_bar_html("Interest Match", int_match, "#00786f") +
            score_bar_html("Overall Fit",    overall,   "#059669") +
            score_bar_html("ROI Score",      min(100, float(roi or 0) * 10), "#f47920"),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="hr-line"></div>', unsafe_allow_html=True)

        # ── Strengths & Concerns ──────────────────────────────
        s_col, c_col = st.columns(2, gap="medium")
        with s_col:
            if strengths:
                s_tags = "".join(f'<span class="tag tag-green"><i class="bi bi-check2"></i> {s}</span>' for s in strengths)
                st.markdown(f"""
                <div class="detail-label"><i class="bi bi-check-circle-fill" style="color:#059669;"></i>&nbsp;Key Strengths</div>
                <div class="tag-row">{s_tags}</div>""", unsafe_allow_html=True)
        with c_col:
            if concerns:
                c_tags = "".join(f'<span class="tag tag-red"><i class="bi bi-exclamation-triangle-fill"></i> {c}</span>' for c in concerns)
                st.markdown(f"""
                <div class="detail-label"><i class="bi bi-exclamation-triangle-fill" style="color:#b91c1c;"></i>&nbsp;Key Concerns</div>
                <div class="tag-row">{c_tags}</div>""", unsafe_allow_html=True)

        # ── Personalised insight ──────────────────────────────
        if insight:
            st.markdown(f"""
            <div class="insight-box">
              <div class="insight-box-label">
                <i class="bi bi-lightbulb-fill"></i> Personalised Insight
              </div>
              {insight}
            </div>""", unsafe_allow_html=True)

        # ── Links ─────────────────────────────────────────────
        if website or (apply_link and apply_link != website):
            link_cols = st.columns(2, gap="small")
            if website:
                link_cols[0].link_button(
                    "Official Website",
                    website,
                    use_container_width=True,
                )
            if apply_link and apply_link != website:
                link_cols[1].link_button(
                    "Apply Now",
                    apply_link,
                    use_container_width=True,
                )


# ════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════════════════════════════════

for key, default in [
    ("colleges", None), ("profile", None),
    ("sources", []),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
      <i class="bi bi-mortarboard-fill"></i> College Finder
    </div>
    <div class="sidebar-tagline">India Edition &nbsp;·&nbsp; Powered by Gemini AI</div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section-label">
      <i class="bi bi-shield-lock-fill"></i> API Status
    </div>""", unsafe_allow_html=True)

    if GEMINI_API_KEY:
        st.markdown("""
        <div class="sidebar-note" style="color:#065f46 !important;">
          <i class="bi bi-check-circle-fill"></i> Connected — ready to search.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-note" style="color:#b91c1c !important;">
          <i class="bi bi-exclamation-triangle-fill"></i>
          No API key configured on the server. The app owner needs to set
          <code>GEMINI_API_KEY</code> in <code>.streamlit/secrets.toml</code>
          or as an environment variable.
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-section-label">
      <i class="bi bi-cpu-fill"></i> Model Selection
    </div>""", unsafe_allow_html=True)

    model_name = st.selectbox(
        "Gemini model",
        MODEL_OPTIONS,
        index=MODEL_OPTIONS.index(DEFAULT_MODEL),
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown(f"""
    <div class="sidebar-note" style="text-align:center;color:#6fa39b !important;">
      AI College Finder &nbsp;·&nbsp; v{APP_VERSION}
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  APP HEADER
# ════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="app-header">
  <div class="app-header-label">
    <i class="bi bi-stars"></i> AI-Powered &nbsp;·&nbsp; Live Google Search &nbsp;·&nbsp; NIRF 2024
  </div>
  <h1>Find Your <span>Perfect College</span></h1>
  <p>
    Complete your profile below. Gemini AI searches real-time data from official college
    websites, NIRF rankings, NAAC ratings, and placement records to deliver
    6 personalised recommendations ranked by fit, ROI, and eligibility.
  </p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
#  PROFILE FORM
# ════════════════════════════════════════════════════════════════════════════

# ── 1. Personal Information ──────────────────────────────────────────────
section_header("bi-person-circle", "Personal Information")
c1, c2 = st.columns(2)
name = c1.text_input("Full name", "Rahul Sharma")
age  = c2.number_input("Age", min_value=15, max_value=30, value=18)

# ── 2. Course Level ──────────────────────────────────────────────────────
section_header("bi-mortarboard-fill", "Course Level")
course_type_key = st.selectbox(
    "Level of study you are pursuing",
    list(COURSE_TYPES.keys()),
    format_func=lambda k: COURSE_TYPES[k],
)

# ── 3. Field of Interest ─────────────────────────────────────────────────
section_header("bi-book-fill", "Field of Interest")
interest_key = st.selectbox(
    "Primary interest area",
    list(INTEREST_AREAS.keys()),
    format_func=lambda k: INTEREST_AREAS[k],
)
specialisation = st.text_input(
    "Specific specialisation or sub-field (optional)",
    placeholder="e.g. Cybersecurity / Cardiac Surgery / Corporate Law",
)

# ── 4. Academic Performance ──────────────────────────────────────────────
section_header("bi-bar-chart-fill", "Academic Performance")
board = class12_score = class10_score = None
graduation_degree = graduation_score = None

if course_type_key in ("1", "3"):
    b1, b2, b3 = st.columns(3)
    board         = b1.text_input("10+2 Board", "CBSE")
    class12_score = b2.number_input("12th Percentage (%)", min_value=33, max_value=100, value=85)
    class10_score = b3.number_input("10th Percentage (%)", min_value=33, max_value=100, value=88)
else:
    g1, g2 = st.columns(2)
    graduation_degree = g1.text_input("Undergraduate Degree", "B.Tech CSE")
    graduation_score  = g2.number_input(
        "UG Score / CGPA (% equivalent)", min_value=33, max_value=100, value=78
    )

# ── 5. Entrance Exam Details ─────────────────────────────────────────────
section_header("bi-clipboard-check-fill", "Entrance Exam Details")
entrance_exam_key = st.selectbox(
    "Entrance exam you have appeared / plan to appear in",
    list(ENTRANCE_EXAMS.keys()),
    format_func=lambda k: ENTRANCE_EXAMS[k],
)
if entrance_exam_key not in ("12", "13"):
    entrance_score = st.text_input(
        f"{ENTRANCE_EXAMS[entrance_exam_key]} — Score / Percentile / Rank",
        placeholder="e.g. 95.4 percentile / AIR 8500 / 680 marks",
    )
else:
    entrance_score = "N/A"

# ── 6. Budget ────────────────────────────────────────────────────────────
section_header("bi-wallet2", "Annual Budget")
budget_key = st.selectbox(
    "Total annual budget (tuition + hostel + living expenses)",
    list(BUDGET_RANGES.keys()),
    format_func=lambda k: BUDGET_RANGES[k][0],
)

# ── 7. Financial Aid & Scholarships ──────────────────────────────────────
section_header("bi-award-fill", "Financial Aid & Scholarships")
scholarship_key = st.selectbox(
    "Your funding plan",
    list(SCHOLARSHIP_OPTIONS.keys()),
    format_func=lambda k: SCHOLARSHIP_OPTIONS[k],
)

# ── 8. Location & Hostel ─────────────────────────────────────────────────
section_header("bi-geo-alt-fill", "Location Preference")
location_keys = st.multiselect(
    "Preferred study location(s)",
    list(LOCATIONS.keys()),
    default=["15"],
    format_func=lambda k: LOCATIONS[k],
)
hostel_key = st.selectbox(
    "Hostel requirement",
    list(HOSTEL_OPTIONS.keys()),
    format_func=lambda k: HOSTEL_OPTIONS[k],
)

# ── 9. College Type ───────────────────────────────────────────────────────
section_header("bi-building-fill", "College Type Preference")
college_type_key = st.selectbox(
    "Preferred college type",
    list(COLLEGE_TYPES.keys()),
    format_func=lambda k: COLLEGE_TYPES[k],
)

# ── 10. Career Goals ─────────────────────────────────────────────────────
section_header("bi-briefcase-fill", "Career Goals & Placement Expectations")
sector_keys = st.multiselect(
    "Target sector(s) after graduation",
    list(PLACEMENT_SECTORS.keys()),
    default=["10"],
    format_func=lambda k: PLACEMENT_SECTORS[k],
)
min_package = st.text_input(
    "Minimum expected starting salary (CTC)",
    placeholder="e.g. ₹6 LPA / ₹10 LPA / Government job",
)

# ── 11. Additional Preferences ───────────────────────────────────────────
section_header("bi-sliders", "Additional Preferences")
extras_keys = st.multiselect(
    "Campus facilities that matter to you",
    list(EXTRACURRICULAR_OPTIONS.keys()),
    default=["8"],
    format_func=lambda k: EXTRACURRICULAR_OPTIONS[k],
)
notes = st.text_area(
    "Any other requirements or notes (optional)",
    placeholder="e.g. Prefer NAAC A++ / need disability support / only AICTE approved",
)

# ── Search Button ─────────────────────────────────────────────────────────
st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
find_clicked = st.button(
    "Find My College Recommendations",
    type="primary",
    use_container_width=True,
)


# ════════════════════════════════════════════════════════════════════════════
#  SEARCH TRIGGER
# ════════════════════════════════════════════════════════════════════════════

if find_clicked:
    if not GEMINI_API_KEY:
        st.error(
            "This app isn't configured with a Gemini API key yet. "
            "The app owner needs to set `GEMINI_API_KEY` in "
            "`.streamlit/secrets.toml` or as an environment variable on the server."
        )
    elif not name.strip():
        st.error("Please enter your full name to personalise recommendations.")
    elif not location_keys:
        st.error("Please select at least one preferred location.")
    elif not sector_keys:
        st.error("Please select at least one target career sector.")
    else:
        profile = {
            "name":             name.strip(),
            "age":              age,
            "course_type":      COURSE_TYPES[course_type_key],
            "course_type_key":  course_type_key,
            "interest":         INTEREST_AREAS[interest_key],
            "interest_key":     interest_key,
            "specialisation":   specialisation.strip() or "None specified",
            "board":            board,
            "class12_score":    class12_score,
            "class10_score":    class10_score,
            "graduation_degree":graduation_degree,
            "graduation_score": graduation_score,
            "entrance_exam":    ENTRANCE_EXAMS[entrance_exam_key],
            "entrance_exam_key":entrance_exam_key,
            "entrance_score":   entrance_score or "N/A",
            "budget_label":     BUDGET_RANGES[budget_key][0],
            "budget_min":       BUDGET_RANGES[budget_key][1],
            "budget_max":       BUDGET_RANGES[budget_key][2],
            "scholarship":      SCHOLARSHIP_OPTIONS[scholarship_key],
            "locations":        [LOCATIONS[k] for k in location_keys],
            "location_keys":    location_keys,
            "hostel":           HOSTEL_OPTIONS[hostel_key],
            "college_type":     COLLEGE_TYPES[college_type_key],
            "target_sectors":   [PLACEMENT_SECTORS[k] for k in sector_keys],
            "min_package":      min_package.strip() or "Not specified",
            "extras":           [EXTRACURRICULAR_OPTIONS[k] for k in extras_keys],
            "notes":            notes.strip() or "None",
        }
        st.session_state.profile = profile
        prompt = build_prompt(profile)

        try:
            with st.spinner("Searching live college data — this typically takes 30–90 seconds…"):
                colleges, sources = call_gemini_with_search(
                    GEMINI_API_KEY, model_name, prompt
                )
            st.session_state.colleges = colleges
            st.session_state.sources  = sources
            st.success(
                f"Search complete — {len(colleges)} college recommendations found for {profile['name']}."
            )
        except Exception as exc:
            st.session_state.colleges = None
            msg = str(exc).lower()
            if any(w in msg for w in ("api_key", "invalid", "authentication", "permission", "401")):
                st.error(
                    "**API Key Error** — the provided Gemini API key is invalid or inactive. "
                    "Please check your key at https://aistudio.google.com/apikey"
                )
            elif any(w in msg for w in ("quota", "rate", "limit", "429")):
                st.error("**Rate limit / quota exceeded.** Please wait a moment and try again.")
            elif any(w in msg for w in ("503", "unavailable", "overloaded")):
                st.error(
                    f"**Gemini server is currently overloaded.** This is a temporary issue on "
                    f"Google's end. Please wait a minute and try again, or switch to the "
                    f"**gemini-2.0-flash** model in the sidebar."
                )
            elif any(w in msg for w in ("not found", "404")):
                st.error(
                    f"**Model not available** — {model_name} is not accessible with your API key. "
                    "Please select a different model in the sidebar."
                )
            elif any(w in msg for w in ("network", "connect", "timeout")):
                st.error("**Network error** — could not reach Google AI servers. Check your connection.")
            else:
                st.error(f"Unexpected error: {exc}")


# ════════════════════════════════════════════════════════════════════════════
#  RESULTS
# ════════════════════════════════════════════════════════════════════════════

if st.session_state.colleges:
    colleges = st.session_state.colleges
    profile  = st.session_state.profile
    budget_max = profile.get("budget_max", 0)

    # ── Results banner ────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="results-banner">
      <div class="results-banner-title">
        <i class="bi bi-mortarboard-fill"></i>
        Recommendations for {profile['name']}
      </div>
      <span class="results-count-pill">{len(colleges)} colleges matched</span>
    </div>""", unsafe_allow_html=True)

    # ── Comparison table ──────────────────────────────────────────────────
    st.markdown("""
    <div class="results-section-title">
      <i class="bi bi-table"></i> Quick Comparison — All Recommendations
    </div>""", unsafe_allow_html=True)

    st.markdown(
        render_comparison_table(colleges, budget_max),
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="font-size:11px;color:#94a3b8;margin-top:-10px;margin-bottom:20px;text-align:right;">
      <i class="bi bi-info-circle"></i>&nbsp;
      Green fee = within budget &nbsp;·&nbsp; Red fee = over budget &nbsp;·&nbsp;
      Scores shown as percentages
    </div>""", unsafe_allow_html=True)

    # ── Detailed profiles ─────────────────────────────────────────────────
    st.markdown("""
    <div class="results-section-title">
      <i class="bi bi-card-list"></i> Detailed College Profiles
    </div>""", unsafe_allow_html=True)

    for college in colleges:
        display_college_detail(college, budget_max)

    # ── Sources ───────────────────────────────────────────────────────────
    if st.session_state.sources:
        with st.expander(
            f"Web Sources Consulted by Gemini ({len(st.session_state.sources)} sources)",
            expanded=False,
        ):
            for s in st.session_state.sources:
                label = s.get("title") or s.get("uri") or "Source"
                uri   = s.get("uri", "")
                if uri:
                    st.markdown(
                        f'<a class="source-link" href="{uri}" target="_blank">'
                        f'<i class="bi bi-link-45deg"></i> {label}</a>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="source-link"><i class="bi bi-link-45deg"></i> {label}</div>',
                        unsafe_allow_html=True,
                    )

    # ── Download ──────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    output = {
        "generated_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
        "app_version":     APP_VERSION,
        "ai_model":        model_name,
        "student_profile": profile,
        "recommendations": colleges,
    }
    st.download_button(
        label="Download Full Report (JSON)",
        data=json.dumps(output, indent=2, ensure_ascii=False),
        file_name=f"college_recommendations_{profile['name'].replace(' ', '_').lower()}.json",
        mime="application/json",
        use_container_width=True,
    )
