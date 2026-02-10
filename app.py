import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
import sqlite3
import nltk
from nltk.tokenize import word_tokenize

# ---------- NLTK ----------
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- DATABASE ----------
conn = sqlite3.connect("datagenie.db", check_same_thread=False)
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

# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ---------- AUTH FUNCTIONS ----------
def register_user(u, p):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
        conn.commit()
        return True
    except:
        return False


def login_user(u, p):
    cursor.execute("SELECT id FROM users WHERE username=? AND password=?", (u, p))
    return cursor.fetchone()


def save_upload(uid, fname):
    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (uid, fname))
    conn.commit()


def get_uploads(uid):
    cursor.execute("SELECT filename, upload_time FROM uploads WHERE user_id=? ORDER BY upload_time DESC", (uid,))
    return cursor.fetchall()

# ---------- LOGIN ----------
def login_page():
    st.title("🤖 DataGenie Login")

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
            st.error("Invalid username or password")

    if st.button("Create new account"):
        st.session_state.page = "register"
        st.rerun()

# ---------- REGISTER ----------
def register_page():
    st.title("📝 Register")

    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Register"):
        if register_user(u, p):
            st.success("Account created. Please login.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------- NLP CHATBOT ----------
def nlp_chatbot(question, df):
    tokens = word_tokenize(question.lower())
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not numeric_cols:
        return "No numeric columns available in dataset."

    col = numeric_cols[0]

    if "total" in tokens:
        return f"Total {col} is {df[col].sum():,.2f}"

    if "average" in tokens or "mean" in tokens:
        return f"Average {col} is {df[col].mean():,.2f}"

    if "max" in tokens or "highest" in tokens:
        return f"Highest {col} is {df[col].max():,.2f}"

    if "predict" in tokens:
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[col].values
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(df)]])[0]
        return f"Next predicted {col} is {pred:,.2f}"

    return "Try asking about total, average, highest, or prediction."

# ---------- PDF GENERATOR ----------
def create_pdf(text, chart_path=None):
    path = "/tmp/datagenie_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()

    elements = []

    for line in text.split("\n"):
        elements.append(Paragraph(line, styles["Normal"]))
        elements.append(Spacer(1, 12))

    if chart_path:
        elements.append(RLImage(chart_path, width=400, height=250))

    doc.build(elements)
    return path

# ---------- MAIN APP ----------
def main_app():
    st.title("📊 DataGenie Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    # ---------- SIDEBAR ----------
    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    st.sidebar.subheader("📜 Upload History")
    for f, t in get_uploads(st.session_state.user_id):
        st.sidebar.write(f"{f} — {t}")

    if file is None:
        st.info("Upload an Excel/CSV file from the sidebar to begin.")
        return

    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)

    save_upload(st.session_state.user_id, file.name)

    # ---------- DATA CLEANING ----------
    df = df.dropna()
    df = df.drop_duplicates()

    # ---------- TABS ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Data Preview",
        "🧹 Data Cleaning",
        "📊 Dashboard",
        "🤖 AI Insights",
        "💬 Chatbot",
    ])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df)

    # ---------- TAB 2 ----------
    with tab2:
        st.subheader("Cleaning Summary")
        st.write("Rows after dropna & duplicates:", df.shape[0])
        st.write("Columns:", list(df.columns))

    # ---------- TAB 3 DASHBOARD ----------
    with tab3:
        st.subheader("Build Dashboard")

        x_col = st.selectbox("X column", df.columns)
        y_col = st.selectbox("Y column", df.select_dtypes(include=np.number).columns)
        chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Pie", "Histogram"])

        fig, ax = plt.subplots()

        if chart_type == "Bar":
            df.groupby(x_col)[y_col].sum().plot(kind="bar", ax=ax)

        elif chart_type == "Line":
            df.groupby(x_col)[y_col].sum().plot(kind="line", ax=ax)

        elif chart_type == "Pie":
            df.groupby(x_col)[y_col].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")

        elif chart_type == "Histogram":
            df[y_col].plot(kind="hist", ax=ax)

        chart_path = "/tmp/chart.png"
        fig.savefig(chart_path)

        st.pyplot(fig)

    # ---------- TAB 4 AI INSIGHTS ----------
    with tab4:
        st.subheader("Detailed AI Insights")

        insights = []
        for col in df.select_dtypes(include=np.number).columns:
            insights.append(
                f"{col} → Total: {df[col].sum():,.2f}, Average: {df[col].mean():,.2f}, Max: {df[col].max():,.2f}"
            )

        insight_text = "\n".join(insights)
        st.text(insight_text)

        if st.button("Download Full PDF Report"):
            pdf = create_pdf(insight_text, chart_path)
            with open(pdf, "rb") as f:
                st.download_button("Download PDF", f, "DataGenie_Report.pdf")

    # ---------- TAB 5 CHATBOT ----------
    with tab5:
        st.subheader("NLP Chatbot")

        q = st.text_input("Ask about your data")
        if q:
            st.success(nlp_chatbot(q, df))

# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()