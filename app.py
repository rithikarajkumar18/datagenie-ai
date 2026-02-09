import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- DATABASE ----------

def get_connection():
    return sqlite3.connect("datagenie.db", check_same_thread=False)

conn = get_connection()
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

# ---------- HELPERS ----------

def get_numeric_columns(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(exclude=np.number).columns.tolist()


# ---------- PDF REPORT ----------

def create_pdf_report(text, filename="report.pdf"):
    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()
    elements = []

    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    doc.build(elements)


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
    st.caption("AI-Powered Decision Support Dashboard")

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

    col1, col2 = st.columns([9, 1])
    with col1:
        st.markdown("<h1>🤖 DataGenie</h1>", unsafe_allow_html=True)
        st.caption("AI-Powered Decision Support Dashboard")

    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    # ---------- SIDEBAR ----------
    st.sidebar.title("📂 Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

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

        df = df.drop_duplicates()
        save_upload(st.session_state.user_id, uploaded_file.name)

    # ---------- NO FILE ----------
    if df is None:
        st.markdown(
            """
            <div style='text-align:center;margin-top:120px;'>
            <h3>📂 No dataset uploaded</h3>
            <p style='color:gray;'>Upload an Excel/CSV file from the sidebar to begin.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        return

    # ---------- COLUMN TYPES ----------
    numeric_cols = get_numeric_columns(df)
    categorical_cols = get_categorical_columns(df)

    tab1, tab2, tab3 = st.tabs(["📄 Data", "📊 Dashboard", "🤖 AI Insights"])

    # ---------- DATA ----------
    with tab1:
        st.dataframe(df, use_container_width=True)

    # ---------- DASHBOARD ----------
    with tab2:

        st.subheader("Bar Chart")
        if numeric_cols and categorical_cols:
            x_col = st.selectbox("Select Category", categorical_cols)
            y_col = st.selectbox("Select Numeric", numeric_cols)

            fig, ax = plt.subplots()
            df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        st.subheader("Pie Chart")
        if categorical_cols:
            pie_col = st.selectbox("Select Column for Pie", categorical_cols, key="pie")

            fig2, ax2 = plt.subplots()
            df[pie_col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
            ax2.set_ylabel("")
            st.pyplot(fig2)

        st.subheader("Line Chart")
        if numeric_cols:
            line_col = st.selectbox("Select Numeric for Trend", numeric_cols, key="line")

            fig3, ax3 = plt.subplots()
            df[line_col].plot(ax=ax3)
            st.pyplot(fig3)

    # ---------- AI INSIGHTS ----------
    with tab3:

        insights = []

        insights.append(f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")

        for col in numeric_cols:
            insights.append(
                f"Column '{col}' → Mean: {df[col].mean():.2f}, Max: {df[col].max():.2f}, Min: {df[col].min():.2f}."
            )

        if categorical_cols:
            for col in categorical_cols[:2]:
                top = df[col].value_counts().idxmax()
                insights.append(f"Most common value in '{col}' is '{top}'.")

        # Prediction using first numeric column
        if numeric_cols:
            col = numeric_cols[0]
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[col].values
            model = LinearRegression().fit(X, y)
            pred = model.predict([[len(df)]])[0]
            insights.append(f"Predicted next value for '{col}' is approximately {pred:.2f}.")

        full_text = "\n".join(insights)

        st.write(full_text)

        if st.button("📄 Download AI Report as PDF"):
            create_pdf_report(full_text)
            with open("report.pdf", "rb") as f:
                st.download_button("Download PDF", f, file_name="DataGenie_Report.pdf")


# ---------- ROUTER ----------

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
