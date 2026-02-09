# DataGenie Advanced App with User-Selectable Charts

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sqlite3
from sklearn.linear_model import LinearRegression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------------- DATABASE ----------------
def get_conn():
    return sqlite3.connect("datagenie.db", check_same_thread=False)

conn = get_conn()
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS uploads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    filename TEXT,
    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# ---------------- SESSION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ---------------- AUTH FUNCTIONS ----------------
def register_user(u, p):
    try:
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
        conn.commit()
        return True
    except:
        return False


def login_user(u, p):
    cur.execute("SELECT id FROM users WHERE username=? AND password=?", (u, p))
    return cur.fetchone()


def save_upload(uid, name):
    cur.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (uid, name))
    conn.commit()


def get_uploads(uid):
    cur.execute("SELECT filename, upload_time FROM uploads WHERE user_id=? ORDER BY upload_time DESC", (uid,))
    return cur.fetchall()

# ---------------- LOGIN PAGE ----------------
def login_page():
    st.title("🤖 DataGenie AI")
    st.subheader("Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(u, p)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.session_state.page = "app"
            st.rerun()
        else:
            st.error("Invalid credentials")

    if st.button("Create Account"):
        st.session_state.page = "register"
        st.rerun()

# ---------------- REGISTER PAGE ----------------
def register_page():
    st.title("📝 Register")

    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Register"):
        if register_user(u, p):
            st.success("Account created. Please login.")
        else:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------------- PDF ----------------
def create_pdf(text):
    path = "/tmp/datagenie_ai_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    elements = []

    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    return path

# ---------------- MAIN APP ----------------
def main_app():

    col1, col2 = st.columns([9, 1])

    with col1:
        st.title("📊 DataGenie Dashboard")

    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    # -------- SIDEBAR --------
    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])

    st.sidebar.markdown("### Previous Uploads")
    uploads = get_uploads(st.session_state.user_id)
    for name, time in uploads:
        st.sidebar.caption(f"{name} — {time}")

    df = None

    if file:
        if file.name.endswith("csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        save_upload(st.session_state.user_id, file.name)

    # -------- NO DATA --------
    if df is None:
        st.info("Upload an Excel or CSV file from the sidebar to begin.")
        return

    # -------- CLEAN --------
    df = df.drop_duplicates()

    # -------- TABS --------
    tab1, tab2, tab3 = st.tabs(["Data", "Dashboard", "AI Insights"])

    # -------- DATA TAB --------
    with tab1:
        st.dataframe(df, use_container_width=True)

    # -------- DASHBOARD TAB --------
    with tab2:
        st.subheader("Create Your Own Chart")

        chart_type = st.selectbox(
            "Choose chart type",
            ["Bar Chart", "Pie Chart", "Line Chart", "Histogram"]
        )

        x_col = st.selectbox("Select X column", df.columns)

        numeric_cols = df.select_dtypes(include=np.number).columns

        y_col = None
        if len(numeric_cols) > 0:
            y_col = st.selectbox("Select Y column", numeric_cols)

        fig, ax = plt.subplots()

        if chart_type == "Bar Chart" and y_col:
            df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)

        elif chart_type == "Pie Chart" and y_col:
            df.groupby(x_col)[y_col].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")

        elif chart_type == "Line Chart" and y_col:
            df.groupby(x_col)[y_col].sum().plot(kind="line", ax=ax)

        elif chart_type == "Histogram":
            df[x_col].plot(kind="hist", ax=ax)

        st.pyplot(fig)

    # -------- AI INSIGHTS TAB --------
    with tab3:
        st.subheader("Automatic AI Insights")

        text = "Dataset Summary:\n"
        text += f"Rows: {df.shape[0]}\nColumns: {df.shape[1]}\n\n"

        num_cols = df.select_dtypes(include=np.number).columns

        for col in num_cols:
            text += f"Column: {col}\n"
            text += f"Average: {df[col].mean():.2f}\n"
            text += f"Max: {df[col].max():.2f}\n"
            text += f"Min: {df[col].min():.2f}\n\n"

        st.text(text)

        if len(num_cols) > 0:
            target = st.selectbox("Select column for prediction", num_cols)

            X = np.arange(len(df)).reshape(-1, 1)
            y = df[target].values

            model = LinearRegression()
            model.fit(X, y)

            pred = model.predict([[len(df)]])[0]
            st.success(f"Next predicted value for {target}: {pred:.2f}")

        if st.button("Download AI Report PDF"):
            pdf = create_pdf(text)
            with open(pdf, "rb") as f:
                st.download_button("Download PDF", f, "AI_Report.pdf")

# ---------------- ROUTER ----------------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
