import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3
import nltk
from nltk.tokenize import word_tokenize

# Download tokenizer once
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

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

# ---------- LOGIN PAGE ----------
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
            st.error("Invalid login")

    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()

# ---------- REGISTER ----------
def register_page():
    st.title("📝 Register")

    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if register_user(u, p):
            st.success("Account created")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username exists")

# ---------- NLP CHATBOT ----------
def nlp_chatbot(question, df):
    tokens = word_tokenize(question.lower())

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if "total" in tokens:
        col = numeric_cols[0] if numeric_cols else None
        if col:
            return f"Total {col} is {df[col].sum():,.2f}"

    if "average" in tokens or "mean" in tokens:
        col = numeric_cols[0] if numeric_cols else None
        if col:
            return f"Average {col} is {df[col].mean():,.2f}"

    if "max" in tokens or "highest" in tokens:
        col = numeric_cols[0] if numeric_cols else None
        if col:
            return f"Highest {col} is {df[col].max():,.2f}"

    if "predict" in tokens and numeric_cols:
        col = numeric_cols[0]
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[col].values
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(df)]])[0]
        return f"Next predicted {col} is {pred:,.2f}"

    return "Sorry, I couldn't understand. Try asking total, average, max, or prediction."

# ---------- MAIN APP ----------
def main_app():
    st.title("📊 DataGenie Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    st.sidebar.header("Upload Dataset")
    file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    # Upload history
    st.sidebar.subheader("📜 Upload History")
    for f, t in get_uploads(st.session_state.user_id):
        st.sidebar.write(f"{f} — {t}")

    if file is None:
        st.info("Upload an Excel/CSV file from the sidebar to begin.")
        return

    # Read file
    df = pd.read_csv(file) if file.name.endswith("csv") else pd.read_excel(file)
    save_upload(st.session_state.user_id, file.name)

    st.subheader("Dataset Preview")
    st.dataframe(df)

    # ---------- CHART BUILDER ----------
    st.subheader("📈 Build Your Chart")

    x_col = st.selectbox("Select X column", df.columns)
    y_col = st.selectbox("Select Y column", df.select_dtypes(include=np.number).columns)
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
        if pd.api.types.is_numeric_dtype(df[y_col]):
            df[y_col].plot(kind="hist", ax=ax)
        else:
            st.warning("Histogram needs numeric column")

    st.pyplot(fig)

    # ---------- AI INSIGHTS ----------
    st.subheader("🤖 AI Insights")

    insights = []
    for col in df.select_dtypes(include=np.number).columns:
        insights.append(f"{col} → Total: {df[col].sum():,.2f}, Avg: {df[col].mean():,.2f}")

    insight_text = "\n".join(insights)
    st.text(insight_text)

    # ---------- PDF ----------
    def create_pdf(text):
        path = "/tmp/report.pdf"
        doc = SimpleDocTemplate(path)
        styles = getSampleStyleSheet()
        elems = []
        for line in text.split("\n"):
            elems.append(Paragraph(line, styles["Normal"]))
            elems.append(Spacer(1, 12))
        doc.build(elems)
        return path

    if st.button("Download AI Report PDF"):
        pdf = create_pdf(insight_text)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f, "AI_Report.pdf")

    # ---------- NLP CHAT ----------
    st.subheader("💬 NLP Chatbot")

    q = st.text_input("Ask about your data")
    if q:
        ans = nlp_chatbot(q, df)
        st.success(ans)

# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()