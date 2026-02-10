import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3
import os
import nltk
from nltk.tokenize import word_tokenize

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- DATABASE ----------
def get_conn():
    return sqlite3.connect("datagenie.db", check_same_thread=False)

conn = get_conn()
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    filename TEXT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

# ---------- USER FUNCTIONS ----------
def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except:
        return False


def check_login(username, password):
    cursor.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()


def save_upload(user_id, filename):
    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (user_id, filename))
    conn.commit()


def get_uploads(user_id):
    cursor.execute("SELECT filename, upload_time FROM uploads WHERE user_id=? ORDER BY upload_time DESC", (user_id,))
    return cursor.fetchall()

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ---------- LOGIN ----------
def login_page():
    st.title("🤖 DataGenie")
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = check_login(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.session_state.page = "app"
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Create account"):
        st.session_state.page = "register"
        st.rerun()

# ---------- REGISTER ----------
def register_page():
    st.title("📝 Register")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Register"):
        if register_user(username, password):
            st.success("Account created. Go to login.")
        else:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------- PDF GENERATOR ----------
def create_pdf(insights_lines, chart_path=None):
    file_path = "datagenie_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("DataGenie AI Report", styles["Title"]))
    elements.append(Spacer(1, 20))

    for line in insights_lines:
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    if chart_path and os.path.exists(chart_path):
        elements.append(Spacer(1, 20))
        elements.append(Paragraph("Dashboard Chart", styles["Heading2"]))
        elements.append(Spacer(1, 12))
        elements.append(RLImage(chart_path, width=400, height=250))

    doc.build(elements)
    return file_path

# ---------- MAIN APP ----------
def main_app():
    st.title("📊 DataGenie Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Upload a dataset to begin.")
        return

    # ---------- LOAD DATA ----------
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    save_upload(st.session_state.user_id, uploaded_file.name)

    # ---------- CLEAN ----------
    df = df.dropna().drop_duplicates()

    # ---------- TABS ----------
    tab1, tab2, tab3 = st.tabs(["Data", "Dashboard", "AI Insights"])

    # ---------- DATA ----------
    with tab1:
        st.dataframe(df)

    # ---------- DASHBOARD ----------
    with tab2:
        chart_type = st.selectbox("Choose chart", ["Bar", "Line", "Pie", "Histogram"])
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns available for chart.")
            return

        col = st.selectbox("Select numeric column", numeric_cols)

        fig, ax = plt.subplots()

        if chart_type == "Bar":
            df[col].value_counts().plot(kind="bar", ax=ax)
        elif chart_type == "Line":
            df[col].plot(ax=ax)
        elif chart_type == "Pie":
            df[col].value_counts().plot(kind="pie", ax=ax, autopct="%1.1f%%")
            ax.set_ylabel("")
        elif chart_type == "Histogram":
            df[col].plot(kind="hist", ax=ax)

        st.pyplot(fig)

        chart_path = "chart.png"
        fig.savefig(chart_path)

    # ---------- AI INSIGHTS ----------
    with tab3:
        insights_lines = []

        for col in numeric_cols:
            insights_lines.append(f"Column: {col}")
            insights_lines.append(f"Total: {df[col].sum():,.2f}")
            insights_lines.append(f"Average: {df[col].mean():,.2f}")
            insights_lines.append(f"Maximum: {df[col].max():,.2f}")
            insights_lines.append(" ")

        # show line by line
        for line in insights_lines:
            st.write(line)

        # ---------- DOWNLOAD FULL REPORT ----------
        if st.button("Download Dashboard + AI Report PDF"):
            pdf_path = create_pdf(insights_lines, chart_path)
            with open(pdf_path, "rb") as f:
                st.download_button("Click to Download PDF", f, "DataGenie_Report.pdf")

# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
