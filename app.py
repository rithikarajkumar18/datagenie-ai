import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from sklearn.linear_model import LinearRegression
import nltk
from nltk.tokenize import word_tokenize
import tempfile
import os

# ──────────────── NLTK SETUP ────────────────
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

# ──────────────── PAGE CONFIG ────────────────
st.set_page_config(
    page_title="DataGenie AI",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# ──────────────── DATABASE SETUP ────────────────
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("datagenie.db", check_same_thread=False)
    return conn

conn = get_db_connection()
cursor = conn.cursor()

cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
""")

cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
""")
conn.commit()

# ──────────────── SESSION STATE INITIALIZATION ────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "df" not in st.session_state:
    st.session_state.df = None

# ──────────────── AUTHENTICATION FUNCTIONS ────────────────
def register_user(username: str, password: str) -> bool:
    try:
        cursor.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username: str, password: str):
    cursor.execute(
        "SELECT id FROM users WHERE username = ? AND password = ?",
        (username, password)
    )
    return cursor.fetchone()

def save_upload(user_id: int, filename: str):
    cursor.execute(
        "INSERT INTO uploads (user_id, filename) VALUES (?, ?)",
        (user_id, filename)
    )
    conn.commit()

def get_uploads(user_id: int):
    cursor.execute(
        "SELECT filename, upload_time FROM uploads WHERE user_id = ? ORDER BY upload_time DESC",
        (user_id,)
    )
    return cursor.fetchall()

# ──────────────── BASIC CLEANING (optional fallback) ────────────────
def clean_data_ui(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("🧹 Quick Data Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Remove rows with missing values"):
            df = df.dropna()
        if st.checkbox("Remove duplicate rows"):
            df = df.drop_duplicates()
    with col2:
        if st.checkbox("Fill numeric missing values with mean"):
            for col in df.select_dtypes(include=np.number).columns:
                df[col] = df[col].fillna(df[col].mean())
    if st.button("Apply Quick Cleaning"):
        st.success("Quick cleaning applied!")
    return df

# ──────────────── ADVANCED CLEANING (main one) ────────────────
def advanced_cleaning_ui(original_df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("⚡ Advanced Data Cleaning")
    st.info("Work on a preview copy → Apply when you're happy with the result.")

    if "working_df" not in st.session_state:
        st.session_state.working_df = original_df.copy()

    df = st.session_state.working_df.copy()

    # ── Missing Values ──
    with st.expander("🩹 1. Missing Values", expanded=True):
        method = st.selectbox(
            "Fill method",
            ["Do nothing", "Mean", "Median", "Mode", "Forward fill", "Backward fill", "0", "Custom"]
        )
        if method == "Custom":
            custom_val = st.text_input("Custom fill value", "")
        if st.button("Apply missing value handling"):
            if method == "Mean":
                df = df.fillna(df.select_dtypes(include=np.number).mean())
            elif method == "Median":
                df = df.fillna(df.select_dtypes(include=np.number).median())
            elif method == "Mode":
                df = df.fillna(df.mode().iloc[0])
            elif method == "Forward fill":
                df = df.ffill()
            elif method == "Backward fill":
                df = df.bfill()
            elif method == "0":
                df = df.fillna(0)
            elif method == "Custom" and custom_val != "":
                df = df.fillna(custom_val)
            st.session_state.working_df = df
            st.success("Missing values handled!")
            st.rerun()

    # ── Duplicates & Outliers ──
    col1, col2 = st.columns(2)
    with col1:
        with st.expander("🗑️ 2. Duplicates"):
            if st.checkbox("Remove duplicate rows"):
                if st.button("Apply duplicate removal"):
                    df = df.drop_duplicates()
                    st.session_state.working_df = df
                    st.success("Duplicates removed!")
                    st.rerun()

    with col2:
        with st.expander("📉 3. Outliers (IQR)"):
            multiplier = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
            if st.button("Remove outliers"):
                num_cols = df.select_dtypes(include=np.number).columns
                for col in num_cols:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower = Q1 - multiplier * IQR
                    upper = Q3 + multiplier * IQR
                    df = df[(df[col] >= lower) & (df[col] <= upper)]
                st.session_state.working_df = df
                st.success("Outliers removed!")
                st.rerun()

    # ── Text Cleaning ──
    with st.expander("✍️ 4. Text Cleaning"):
        text_cols = df.select_dtypes(include="object").columns.tolist()
        selected_text_cols = st.multiselect("Columns to clean", text_cols, default=text_cols[:3])

        trim = st.checkbox("Trim whitespace", value=True)
        lower = st.checkbox("To lowercase")
        title = st.checkbox("To title case")
        remove_special = st.checkbox("Remove special chars (keep alphanum + space)")

        if st.button("Apply text cleaning"):
            for col in selected_text_cols:
                if trim:
                    df[col] = df[col].str.strip()
                if lower:
                    df[col] = df[col].str.lower()
                if title:
                    df[col] = df[col].str.title()
                if remove_special:
                    df[col] = df[col].str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
            st.session_state.working_df = df
            st.success("Text cleaning applied!")
            st.rerun()

    # ── Column Rename / Type Conversion ──
    colA, colB = st.columns(2)
    with colA:
        with st.expander("✏️ 5. Rename Column"):
            old_col = st.selectbox("Select column", df.columns, key="rename_old_col")
            new_col_name = st.text_input("New name", value=old_col)
            if st.button("Rename") and new_col_name.strip() != old_col:
                df.rename(columns={old_col: new_col_name.strip()}, inplace=True)
                st.session_state.working_df = df
                st.success(f"Renamed → {new_col_name}")
                st.rerun()

    with colB:
        with st.expander("🔄 6. Change Data Type"):
            col_to_convert = st.selectbox("Column", df.columns, key="dtype_col")
            target_type = st.selectbox("Convert to", ["int", "float", "str", "datetime", "category"])
            if st.button("Convert type"):
                try:
                    if target_type == "datetime":
                        df[col_to_convert] = pd.to_datetime(df[col_to_convert], errors="coerce")
                    else:
                        df[col_to_convert] = df[col_to_convert].astype(target_type)
                    st.session_state.working_df = df
                    st.success(f"Converted {col_to_convert} → {target_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Conversion failed: {str(e)}")

    # ── Reset & Final Apply ──
    st.divider()
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        if st.button("✅ Apply All Changes & Close Editor", type="primary"):
            st.session_state.df = st.session_state.working_df.copy()
            st.success("Advanced cleaning applied to main dataset!")
            st.session_state.pop("working_df", None)
            st.rerun()
    with c2:
        if st.button("⟳ Reset to Original Upload"):
            st.session_state.working_df = original_df.copy()
            st.success("Reset complete")
            st.rerun()
    with c3:
        st.metric("Rows", len(df))

    st.subheader("Preview of current working data")
    st.dataframe(df.head(12))

    return df

# ──────────────── SIMPLE NLP CHATBOT ────────────────
def nlp_chatbot(question: str, df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "No data loaded yet."

    tokens = word_tokenize(question.lower())
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if not num_cols:
        return "No numeric columns found in the dataset."

    # Very basic keyword matching
    main_col = num_cols[0]  # naive – improve later

    if any(w in tokens for w in ["total", "sum"]):
        return f"Total of {main_col}: **{df[main_col].sum():,.2f}**"

    if any(w in tokens for w in ["average", "avg", "mean"]):
        return f"Average of {main_col}: **{df[main_col].mean():,.2f}**"

    if any(w in tokens for w in ["max", "highest", "maximum"]):
        return f"Highest {main_col}: **{df[main_col].max():,.2f}**"

    if "predict" in tokens or "next" in tokens:
        try:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[main_col].values
            model = LinearRegression().fit(X, y)
            next_val = model.predict([[len(df)]])[0]
            return f"Naive linear prediction for next {main_col}: **{next_val:,.2f}**"
        except:
            return "Could not run prediction (data issue)."

    return "I understand questions like:\n• total / sum\n• average / mean\n• max / highest\n• predict next"

# ──────────────── PDF REPORT ────────────────
def create_report_pdf(insights_text: str, chart_path: str = None) -> str:
    pdf_path = os.path.join(tempfile.gettempdir(), "DataGenie_Report.pdf")

    try:
        pdfmetrics.registerFont(UnicodeCIDFont("Helvetica"))  # fallback if HYSMyeongJo not available
    except:
        pass

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("DataGenie AI Report", styles["Heading1"]))
    elements.append(Spacer(1, 12))

    for line in insights_text.split("\n"):
        if line.strip():
            elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 8))

    if chart_path and os.path.exists(chart_path):
        try:
            elements.append(Image(chart_path, width=480, height=320))
        except:
            elements.append(Paragraph("[Chart image could not be included]", styles["Italic"]))

    doc.build(elements)
    return pdf_path

# ──────────────── LOGIN PAGE ────────────────
def login_page():
    st.title("🔐 DataGenie Login")
    st.markdown("Welcome back! Please sign in.")

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            user = login_user(username, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user_id = user[0]
                st.session_state.page = "app"
                st.rerun()
            else:
                st.error("Invalid username or password")

        if st.button("Create new account", use_container_width=True):
            st.session_state.page = "register"
            st.rerun()

# ──────────────── REGISTER PAGE ────────────────
def register_page():
    st.title("📝 Create Account")
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Choose username")
        password = st.text_input("Choose password", type="password")
        password2 = st.text_input("Confirm password", type="password")

        if st.button("Register", use_container_width=True):
            if password != password2:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password should be at least 6 characters")
            elif register_user(username, password):
                st.success("Account created! Please login.")
                st.session_state.page = "login"
                st.rerun()
            else:
                st.error("Username already taken")

        if st.button("← Back to Login"):
            st.session_state.page = "login"
            st.rerun()

# ──────────────── MAIN APPLICATION ────────────────
def main_app():
    st.title("📊 DataGenie AI Dashboard")
    if st.button("Logout", type="secondary"):
        for key in list(st.session_state.keys()):
            if key not in ["logged_in", "page"]:
                del st.session_state[key]
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    # Sidebar
    with st.sidebar:
        st.header("📁 Upload Data")
        uploaded_file = st.file_uploader("CSV or Excel", type=["csv", "xlsx"])

        st.subheader("Recent Uploads")
        uploads = get_uploads(st.session_state.user_id)
        for fname, ts in uploads[:5]:
            st.caption(f"{fname} • {ts}")

    # Process upload
    if uploaded_file is not None and st.session_state.df is None:
        try:
            if uploaded_file.name.endswith(".csv"):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            save_upload(st.session_state.user_id, uploaded_file.name)
            st.success(f"Loaded: {uploaded_file.name}")
            st.rerun()
        except Exception as e:
            st.error(f"Could not read file: {e}")

    if st.session_state.df is None:
        st.info("Please upload a CSV or Excel file to start analyzing.")
        return

    df = st.session_state.df

    # TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧼 Clean Data",
        "👀 Preview",
        "📈 Visualize",
        "🔍 Insights",
        "💬 Ask AI"
    ])

    with tab1:
        st.session_state.df = advanced_cleaning_ui(df)
        # Optionally keep old quick clean too:
        # st.markdown("---")
        # st.session_state.df = clean_data_ui(st.session_state.df)

    with tab2:
        st.subheader("Current Dataset")
        st.dataframe(st.session_state.df, use_container_width=True)

    with tab3:
        st.subheader("Create Chart")

        if len(df.columns) < 2:
            st.warning("Need at least 2 columns to plot.")
        else:
            x = st.selectbox("X-axis", df.columns)
            y_candidates = df.select_dtypes(include=np.number).columns.tolist()
            y = st.selectbox("Y-axis (numeric)", y_candidates if y_candidates else df.columns)
            chart_type = st.selectbox("Chart type", ["Bar", "Line", "Area", "Pie", "Histogram"])

            fig, ax = plt.subplots(figsize=(10, 6))

            try:
                if chart_type == "Bar":
                    df.groupby(x)[y].sum().plot.bar(ax=ax)
                elif chart_type == "Line":
                    df.groupby(x)[y].sum().plot.line(ax=ax)
                elif chart_type == "Area":
                    df.groupby(x)[y].sum().plot.area(ax=ax)
                elif chart_type == "Pie":
                    df.groupby(x)[y].sum().plot.pie(autopct="%1.1f%%", ax=ax)
                    ax.set_ylabel("")
                elif chart_type == "Histogram":
                    df[y].plot.hist(ax=ax, bins=25)

                plt.tight_layout()
                st.pyplot(fig)

                # Save for PDF
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
                    st.session_state.chart_path = tmp.name

            except Exception as e:
                st.error(f"Chart could not be created: {e}")

    with tab4:
        st.subheader("Quick Business Insights")

        num_cols = df.select_dtypes(include=np.number).columns
        if len(num_cols) == 0:
            st.info("No numeric columns found.")
        else:
            insights = []
            for col in num_cols[:6]:  # limit to avoid huge report
                insights.append(
                    f"**{col}**  •  Total: {df[col].sum():,.1f}  •  Avg: {df[col].mean():,.2f}  •  Max: {df[col].max():,.2f}"
                )

            insight_text = "\n\n".join(insights)

            if num_cols.size > 0:
                pred_col = num_cols[0]
                try:
                    X = np.arange(len(df)).reshape(-1, 1)
                    y = df[pred_col].fillna(0).values
                    model = LinearRegression().fit(X, y)
                    next_pred = model.predict([[len(df)]])[0]
                    insight_text += f"\n\n**Prediction**: Next {pred_col} ≈ {next_pred:,.2f} (simple linear trend)"
                except:
                    pass

            st.markdown(insight_text)

            if st.button("Download PDF Report"):
                pdf_path = create_report_pdf(insight_text, st.session_state.get("chart_path"))
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Download Report PDF",
                        data=f,
                        file_name="DataGenie_Report.pdf",
                        mime="application/pdf"
                    )

    with tab5:
        st.subheader("Chat with your Data (basic)")
        question = st.text_input("Ask something about the data…", key="chat_input")
        if question:
            with st.spinner("Thinking..."):
                answer = nlp_chatbot(question, df)
            st.markdown(f"**→** {answer}")

# ──────────────── ROUTER ────────────────
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
