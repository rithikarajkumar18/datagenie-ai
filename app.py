import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3
import os

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- DATABASE ----------
conn = sqlite3.connect("/mount/src/datagenie-ai/datagenie.db", check_same_thread=False)
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

# ---------- AUTH ----------
def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Register error: {e}")
        return False
    


def login_user(username, password):
    cursor.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
    return cursor.fetchone()


def save_upload(user_id, filename):
    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (user_id, filename))
    conn.commit()


def get_uploads(user_id):
    cursor.execute("SELECT filename, upload_time FROM uploads WHERE user_id=? ORDER BY upload_time DESC", (user_id,))
    return cursor.fetchall()

# ---------- DATA CLEANING (USER CONTROLLED) ----------
def clean_data_ui(df):
    st.subheader("🧹 Data Cleaning Options")

    original_shape = df.shape

    remove_empty_rows = st.checkbox("Remove rows with empty values", key="clean_rows")
    remove_empty_cols = st.checkbox("Remove columns with empty values", key="clean_cols")
    remove_duplicates = st.checkbox("Remove duplicate rows", key="clean_dup")
    fill_missing = st.checkbox("Fill missing numeric values with mean", key="clean_fill")
    

    if st.button("Apply Cleaning", key="apply_cleaning"):

        if remove_empty_rows:
            df = df.dropna(axis=0)

        if remove_empty_cols:
            df = df.dropna(axis=1)

        if remove_duplicates:
            df = df.drop_duplicates()

        if fill_missing:
            for col in df.select_dtypes(include=np.number).columns:
                df[col] = df[col].fillna(df[col].mean())

        st.success("Cleaning applied successfully!")

        st.write("**Rows before cleaning:**", original_shape[0])
        st.write("**Rows after cleaning:**", df.shape[0])

    return df

# ---------- NLP ----------
def nlp_chatbot(question, df):
    tokens = question.lower().split()
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    if "total" in tokens and numeric_cols:
        col = numeric_cols[0]
        return f"Total {col} is {df[col].sum():,.2f}"

    if ("average" in tokens or "mean" in tokens) and numeric_cols:
        col = numeric_cols[0]
        return f"Average {col} is {df[col].mean():,.2f}"

    if ("max" in tokens or "highest" in tokens) and numeric_cols:
        col = numeric_cols[0]
        return f"Highest {col} is {df[col].max():,.2f}"

    if "predict" in tokens and numeric_cols:
        col = numeric_cols[0]
        X = np.arange(len(df)).reshape(-1, 1)
        y = df[col].values
        model = LinearRegression().fit(X, y)
        pred = model.predict([[len(df)]])[0]
        return f"Next predicted {col} is {pred:,.2f}"

    return "Try asking about total, average, highest, or prediction."

# ---------- LOGIN ----------
def login_page():
    st.title("🤖 DataGenie Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.session_state.page = "app"
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()

# ---------- REGISTER ----------
def register_page():
    st.title("📝 Register")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Create Account"):
        if register_user(username, password):
            st.success("Account created. Please login.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username already exists")

# ---------- PDF ----------
def create_full_pdf(text, fig):
    path = "/tmp/datagenie_full_report.pdf"
    doc = SimpleDocTemplate(path)
    styles = getSampleStyleSheet()
    elems = []

    for line in text.split("\n"):
        elems.append(Paragraph(line, styles["Normal"]))
        elems.append(Spacer(1, 12))

    if fig:
        img_path = "/tmp/chart.png"
        fig.savefig(img_path)
        elems.append(Image(img_path, width=400, height=300))

    doc.build(elems)
    return path

# ---------- MAIN ----------
def main_app():
    st.title("📊 DataGenie Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    st.sidebar.header("Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

    st.sidebar.subheader("📜 Upload History")
    for fname, time in get_uploads(st.session_state.user_id):
        st.sidebar.write(f"{fname} — {time}")

    if uploaded_file is None:
        st.info("Upload an Excel/CSV file from the sidebar to begin.")
        return

    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    df = clean_data_ui(df)
    save_upload(st.session_state.user_id, uploaded_file.name)

    tab1, tab2, tab3, tab4 = st.tabs([
    "🧹 Data Cleaning",
    "📊 Dashboard",
    "🤖 AI Insights",
    "💬 Chatbot",
    ])

    # ---------- TAB 1 ----------
    with tab1:
        st.subheader("Cleaned Dataset")
        st.dataframe(df, use_container_width=True)
        x_col = st.selectbox("Select X column", df.columns)
        numeric_cols = df.select_dtypes(include=np.number).columns

        if len(numeric_cols) > 0:
            y_col = st.selectbox("Select Y column", numeric_cols)
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

            st.pyplot(fig)
            st.session_state["last_fig"] = fig

    # ---------- TAB 2 ----------
    with tab2:
        important_keywords = ["sales", "profit", "revenue", "amount", "quantity", "cost"]

        important_cols = [
            col for col in df.select_dtypes(include=np.number).columns
            if any(key in col.lower() for key in important_keywords)
        ]

        insights = []

        if important_cols:
            for col in important_cols:
                insights.append(
                    f"{col} → Total: {df[col].sum():,.2f}, Avg: {df[col].mean():,.2f}, Max: {df[col].max():,.2f}"
                )
        else:
            insights.append("No major business metric columns found.")

        insight_text = "\n".join(insights)
        st.text(insight_text)

        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            col = numeric_cols[0]

            X = np.arange(len(df)).reshape(-1, 1)
            y = df[col].values
            model = LinearRegression().fit(X, y)
            pred = model.predict([[len(df)]])[0]

            st.success(f"Predicted next {col}: {pred:,.2f}")

        fig = st.session_state.get("last_fig", None)

        if st.button("Download Full Report (Dashboard + AI)"):
            pdf_path = create_full_pdf(insight_text, fig)
            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, "DataGenie_Report.pdf")

    # ---------- TAB 3 ----------
    with tab3:
        question = st.text_input("Ask your question")

        if question:
            answer = nlp_chatbot(question, df)
            st.success(answer)

# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
