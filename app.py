import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np
import sqlite3

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------------- DATABASE ----------------

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

# ---------------- SESSION ----------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if "user_id" not in st.session_state:
    st.session_state.user_id = None

# ---------------- LOGIN ----------------

def login_page():
    st.title("🤖 DataGenie")
    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute("SELECT id FROM users WHERE username=? AND password=?", (username, password))
        user = cursor.fetchone()

        if user:
            st.session_state.logged_in = True
            st.session_state.user_id = user[0]
            st.session_state.page = "app"
            st.rerun()
        else:
            st.error("Invalid login")

    if st.button("Create account"):
        st.session_state.page = "register"
        st.rerun()

# ---------------- REGISTER ----------------

def register_page():
    st.title("📝 Register")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password", type="password")

    if st.button("Register"):
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (new_user, new_pass))
            conn.commit()
            st.success("Registered! Go login.")
        except:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ---------------- MAIN APP ----------------

def main_app():

    col1, col2 = st.columns([9, 1])
    with col1:
        st.title("🤖 DataGenie Dashboard")
    with col2:
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

    uploaded_file = st.sidebar.file_uploader("Upload Excel/CSV", type=["csv", "xlsx"])

    if uploaded_file is None:
        st.info("Upload a dataset from sidebar to continue.")
        return

    # -------- READ FILE --------

    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # -------- SAVE HISTORY --------

    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)",
                   (st.session_state.user_id, uploaded_file.name))
    conn.commit()

    # -------- CLEANING --------

    df = df.dropna()
    df = df.drop_duplicates()

    # -------- TABS --------

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Data",
        "🧹 Cleaning",
        "📊 Dashboard",
        "🤖 AI Insights",
        "💬 Chatbot",
    ])

    # -------- TAB 1 DATA --------

    with tab1:
        st.dataframe(df, use_container_width=True)

    # -------- TAB 2 CLEANING --------

    with tab2:
        st.write("Rows after cleaning:", df.shape[0])
        st.write("Columns:", df.shape[1])
        st.success("Empty rows removed using dropna().")

    # -------- TAB 3 DASHBOARD --------

    with tab3:
        st.subheader("Create your chart")

        all_cols = df.columns.tolist()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        chart_type = st.selectbox("Chart type", ["Bar", "Line", "Pie", "Histogram"])

        x_col = st.selectbox("Select column", all_cols)

        fig, ax = plt.subplots()

        try:
            if chart_type == "Histogram":
                if x_col in numeric_cols:
                    df[x_col].plot(kind="hist", ax=ax)
                else:
                    st.error("Histogram needs numeric column")
                    st.stop()

            elif chart_type == "Pie":
                df[x_col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")

            elif chart_type == "Bar":
                df[x_col].value_counts().plot(kind="bar", ax=ax)

            elif chart_type == "Line":
                if x_col in numeric_cols:
                    df[x_col].plot(kind="line", ax=ax)
                else:
                    st.error("Line chart needs numeric column")
                    st.stop()

            st.pyplot(fig)
            chart_path = "chart.png"
            fig.savefig(chart_path)
            st.session_state.chart_path = chart_path

        except Exception as e:
            st.error("Chart error: choose correct column type")

    # -------- TAB 4 AI INSIGHTS --------

    with tab4:

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if not numeric_cols:
            st.warning("No numeric columns for AI insights")
            return

        col = numeric_cols[0]

        total = df[col].sum()
        avg = df[col].mean()
        max_val = df[col].max()

        insights = [
            f"Column used for analysis: {col}",
            f"Total value: {total:.2f}",
            f"Average value: {avg:.2f}",
            f"Maximum value: {max_val:.2f}",
            "Recommendation: Focus on improving high-performing segments.",
        ]

        for line in insights:
            st.write("•", line)

        # -------- PDF DOWNLOAD --------

        def create_pdf():
            path = "report.pdf"
            doc = SimpleDocTemplate(path)
            styles = getSampleStyleSheet()
            elements = []

            for line in insights:
                elements.append(Paragraph(line, styles["Normal"]))
                elements.append(Spacer(1, 12))

            if "chart_path" in st.session_state:
                elements.append(Image(st.session_state.chart_path, width=400, height=300))

            doc.build(elements)
            return path

        if st.button("Download Full Report PDF"):
            pdf = create_pdf()
            with open(pdf, "rb") as f:
                st.download_button("Download PDF", f, "DataGenie_Report.pdf")

    # -------- TAB 5 CHATBOT --------

    with tab5:
        st.subheader("Ask about your data")

        question = st.text_input("Type your question")

        if question:
            q = question.lower()
            col = df.select_dtypes(include=np.number).columns[0]

            if "total" in q:
                st.success(f"Total {col} is {df[col].sum():.2f}")
            elif "average" in q or "mean" in q:
                st.success(f"Average {col} is {df[col].mean():.2f}")
            elif "max" in q or "highest" in q:
                st.success(f"Maximum {col} is {df[col].max():.2f}")
            else:
                st.info("Try asking: total, average, max")

# ---------------- ROUTER ----------------

if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
