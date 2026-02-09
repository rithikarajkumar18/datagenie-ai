import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3
from datetime import datetime

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- DATABASE ----------
def get_connection():
    return sqlite3.connect("datagenie.db", check_same_thread=False)

conn = get_connection()
cursor = conn.cursor()

# Create tables if not exist
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
    upload_time TEXT
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
    cursor.execute(
        "INSERT INTO uploads (user_id, filename, upload_time) VALUES (?, ?, ?)",
        (user_id, filename, datetime.now().strftime("%d %b %Y %H:%M"))
    )
    conn.commit()


def get_uploads(user_id):
    cursor.execute(
        "SELECT filename, upload_time FROM uploads WHERE user_id=? ORDER BY id DESC",
        (user_id,)
    )
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
    st.markdown("<h1 style='text-align:center;'>🤖 DataGenie</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray;'>AI-Powered Decision Support Dashboard</p>", unsafe_allow_html=True)

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
            st.error("Invalid username or password")

    if st.button("Create new account"):
        st.session_state.page = "register"
        st.rerun()


# ---------- REGISTER ----------
def register_page():
    st.markdown("<h1 style='text-align:center;'>📝 Register</h1>", unsafe_allow_html=True)

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    confirm = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if new_pass != confirm:
            st.error("Passwords do not match")
        elif register_user(new_user, new_pass):
            st.success("Account created! Go to login.")
        else:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


# ---------- MAIN APP ----------
def main_app():

    # HEADER
    col1, col2 = st.columns([9, 1])
    with col1:
        st.markdown("<h1>🤖 DataGenie</h1>", unsafe_allow_html=True)
        st.caption("AI-Powered Decision Support Dashboard")

    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    # SIDEBAR UPLOAD
    st.sidebar.title("📂 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    # SHOW HISTORY
    st.sidebar.markdown("### 🕘 Previous Uploads")
    uploads = get_uploads(st.session_state.user_id)
    for f, t in uploads:
        st.sidebar.write(f"**{f}**  \n{t}")

    df = None

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        save_upload(st.session_state.user_id, uploaded_file.name)

    # ---------- IF DATA EXISTS ----------
    if df is not None:

        df = df.drop_duplicates()

        # KPIs
        if "Sales" in df.columns:
            colA, colB, colC = st.columns(3)
            colA.metric("Total Sales", f"{df['Sales'].sum():,.0f}")
            colB.metric("Average Sales", f"{df['Sales'].mean():,.0f}")
            colC.metric("Max Sale", f"{df['Sales'].max():,.0f}")

        st.divider()

        tab1, tab2, tab3, tab4 = st.tabs(["📄 Data", "📊 Dashboard", "🤖 AI Insights", "💬 Chatbot"])

        # DATA TAB
        with tab1:
            st.dataframe(df, use_container_width=True)

        # DASHBOARD TAB
        with tab2:
            if "Sales" in df.columns:

                fig, ax = plt.subplots()
                df["Sales"].plot(ax=ax)
                st.pyplot(fig)

        # AI TAB
        with tab3:
            if "Sales" in df.columns:
                total = df["Sales"].sum()
                avg = df["Sales"].mean()

                st.info(f"Total Sales: {total:,.2f}\n\nAverage Sales: {avg:,.2f}")

                X = np.arange(len(df)).reshape(-1, 1)
                y = df["Sales"].values
                model = LinearRegression().fit(X, y)
                pred = model.predict([[len(df)]])[0]

                st.success(f"Next predicted sale: {pred:,.2f}")

        # CHAT TAB
        with tab4:
            q = st.text_input("Ask about sales...")
            if q and "Sales" in df.columns:
                if "total" in q.lower():
                    st.success(df["Sales"].sum())
                elif "average" in q.lower():
                    st.success(df["Sales"].mean())

    # ---------- NO FILE ----------
    else:
        st.markdown(
            """
            <div style='text-align:center;margin-top:120px;'>
            <h3>📂 No dataset uploaded</h3>
            <p style='color:gray;'>Upload an Excel/CSV file from the sidebar to begin.</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
