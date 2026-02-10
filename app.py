import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import re

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
    filename TEXT
)
""")
conn.commit()


# ---------- SESSION ----------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "page" not in st.session_state:
    st.session_state.page = "login"


# ---------- AUTH ----------
def login_page():
    st.title("🤖 DataGenie")
    st.subheader("Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        cursor.execute("SELECT id FROM users WHERE username=? AND password=?", (u, p))
        user = cursor.fetchone()
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


def register_page():
    st.title("📝 Register")

    u = st.text_input("New Username")
    p = st.text_input("New Password", type="password")

    if st.button("Register"):
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (u, p))
            conn.commit()
            st.success("Registered! Go to login.")
        except:
            st.error("Username already exists")

    if st.button("Back to login"):
        st.session_state.page = "login"
        st.rerun()


# ---------- NLP CHATBOT ----------
def nlp_answer(question, df):
    q = question.lower()

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    detected_col = None
    for col in numeric_cols:
        if col.lower() in q:
            detected_col = col
            break

    if detected_col is None:
        detected_col = numeric_cols[0]

    if re.search(r"total|sum", q):
        return f"Total of {detected_col} is {df[detected_col].sum():.2f}"

    if re.search(r"average|mean", q):
        return f"Average of {detected_col} is {df[detected_col].mean():.2f}"

    if re.search(r"max|highest", q):
        return f"Maximum of {detected_col} is {df[detected_col].max():.2f}"

    if re.search(r"min|lowest", q):
        return f"Minimum of {detected_col} is {df[detected_col].min():.2f}"

    return "Ask about total, average, max, or min of numeric columns."


# ---------- MAIN APP ----------
def main_app():
    st.title("📊 DataGenie Dashboard")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    uploaded = st.sidebar.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"])

    if uploaded is None:
        st.info("Upload a dataset from the sidebar to begin.")
        return

    if uploaded.name.endswith("csv"):
        df = pd.read_csv(uploaded)
    else:
        df = pd.read_excel(uploaded)

    # ---------- CLEANING ----------
    df = df.dropna()
    df = df.drop_duplicates()

    # ---------- SAVE UPLOAD ----------
    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (st.session_state.user_id, uploaded.name))
    conn.commit()

    # ---------- TABS ----------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📄 Data",
        "🧹 Cleaning",
        "📊 Dashboard",
        "🤖 AI Insights",
        "💬 Chatbot",
    ])

    # ---------- DATA ----------
    with tab1:
        st.dataframe(df)

    # ---------- CLEANING ----------
    with tab2:
        st.write("Rows after cleaning:", df.shape[0])
        st.write("Columns:", df.shape[1])

    # ---------- DASHBOARD ----------
    with tab3:
        chart = st.selectbox("Chart type", ["Bar", "Line", "Pie", "Histogram"])

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        all_cols = df.columns.tolist()

        if chart in ["Line", "Histogram"]:
            col = st.selectbox("Select numeric column", numeric_cols)
        else:
            col = st.selectbox("Select column", all_cols)

        fig, ax = plt.subplots()

        if chart == "Bar":
            df[col].value_counts().plot(kind="bar", ax=ax)
        elif chart == "Line":
            df[col].plot(kind="line", ax=ax)
        elif chart == "Pie":
            df[col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
        elif chart == "Histogram":
            df[col].plot(kind="hist", ax=ax)

        st.pyplot(fig)

    # ---------- AI INSIGHTS ----------
    with tab4:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        insight_col = st.selectbox("Column for analysis", numeric_cols)

        total_val = df[insight_col].sum()
        avg_val = df[insight_col].mean()
        max_val = df[insight_col].max()

        st.write(f"• Column: {insight_col}")
        st.write(f"• Total: {total_val:.2f}")
        st.write(f"• Average: {avg_val:.2f}")
        st.write(f"• Maximum: {max_val:.2f}")
        st.write("• Recommendation: Focus on high-performing segments.")

    # ---------- CHATBOT ----------
    with tab5:
        q = st.text_input("Ask about your data")
        if q:
            st.success(nlp_answer(q, df))


# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()