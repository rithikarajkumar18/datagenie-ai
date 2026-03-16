import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sqlite3
import os
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import nltk

# ──────────────── NLTK DATA SETUP ────────────────
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True, download_dir=nltk_data_dir)

# ──────────────── PAGE CONFIG ────────────────
st.set_page_config(
    page_title="DataGenie AI",
    layout="wide",
    page_icon="🪄",
    initial_sidebar_state="expanded"
)

# ──────────────── DATABASE (fixed path) ────────────────
@st.cache_resource
def get_db_connection():
    # Force database to live in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "datagenie.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
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

# ──────────────── SESSION STATE ────────────────
defaults = {
    "logged_in": False,
    "page": "login",
    "user_id": None,
    "df": None,
    "df_filename": None,
    "working_df": None,
    "chart_path": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ──────────────── AUTH FUNCTIONS ────────────────
def register_user(username, password):
    try:
        cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def login_user(username, password):
    cursor.execute("SELECT id FROM users WHERE username = ? AND password = ?", (username, password))
    return cursor.fetchone()

def save_upload(user_id, filename):
    cursor.execute("INSERT INTO uploads (user_id, filename) VALUES (?, ?)", (user_id, filename))
    conn.commit()

def get_uploads(user_id):
    cursor.execute("SELECT filename, upload_time FROM uploads WHERE user_id = ? ORDER BY upload_time DESC", (user_id,))
    return cursor.fetchall()

# ──────────────── ADVANCED CLEANING ────────────────
def advanced_cleaning_ui(original_df):
    if original_df is None or original_df.empty:
        st.error("No valid dataframe passed to cleaning function")
        return original_df

    if st.session_state.working_df is None:
        st.session_state.working_df = original_df.copy()

    df = st.session_state.working_df
    st.subheader("⚡ Advanced Data Cleaning Studio")
    st.info("Preview changes → Apply when ready")

    # Missing values
    with st.expander("1. Missing Values", True):
        method = st.selectbox("Fill method", [
            "Do nothing", "Mean", "Median", "Mode", "Forward fill", "Backward fill", "0", "Custom"
        ])
        custom_val = ""
        if method == "Custom":
            custom_val = st.text_input("Fill with", "")
        if st.button("Apply missing value fix"):
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
            elif method == "Custom" and custom_val:
                df = df.fillna(custom_val)
            st.session_state.working_df = df
            st.success("Missing values handled")
            st.rerun()

    col_left, col_right = st.columns(2)
    with col_left:
        with st.expander("2. Duplicates"):
            if st.button("Remove duplicate rows"):
                df = df.drop_duplicates()
                st.session_state.working_df = df
                st.success("Duplicates removed")
                st.rerun()

    with col_right:
        with st.expander("3. Outliers"):
            mult = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.1)
            if st.button("Remove outliers"):
                num = df.select_dtypes(include=np.number).columns
                for col in num:
                    Q1, Q3 = df[col].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    df = df[(df[col] >= Q1 - mult*IQR) & (df[col] <= Q3 + mult*IQR)]
                st.session_state.working_df = df
                st.success("Outliers removed")
                st.rerun()

    with st.expander("4. Text Cleaning"):
        text_cols = df.select_dtypes("object").columns.tolist()
        sel_cols = st.multiselect("Clean these columns", text_cols, text_cols[:min(3, len(text_cols))])
        trim = st.checkbox("Trim spaces", True)
        lower = st.checkbox("Lowercase")
        title = st.checkbox("Title Case")
        no_special = st.checkbox("Remove special chars")
        if st.button("Apply text cleaning"):
            for col in sel_cols:
                s = df[col].astype(str)
                if trim: s = s.str.strip()
                if lower: s = s.str.lower()
                if title: s = s.str.title()
                if no_special: s = s.str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
                df[col] = s
            st.session_state.working_df = df
            st.success("Text cleaned")
            st.rerun()

    colA, colB = st.columns(2)
    with colA:
        with st.expander("5. Rename Column"):
            old = st.selectbox("Column", df.columns, key="rename_sel")
            new_name = st.text_input("New name", old)
            if st.button("Rename") and new_name.strip() != old:
                df = df.rename(columns={old: new_name.strip()})
                st.session_state.working_df = df
                st.success(f"→ {new_name}")
                st.rerun()

    with colB:
        with st.expander("6. Change Type"):
            col = st.selectbox("Column", df.columns, key="type_sel")
            to_type = st.selectbox("To", ["int", "float", "str", "datetime", "category"])
            if st.button("Convert"):
                try:
                    if to_type == "datetime":
                        df[col] = pd.to_datetime(df[col], errors="coerce")
                    else:
                        df[col] = df[col].astype(to_type)
                    st.session_state.working_df = df
                    st.success(f"{col} → {to_type}")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")

    st.divider()
    c1, c2 = st.columns([3,1])
    with c1:
        if st.button("✅ Apply All Changes", type="primary"):
            st.session_state.df = st.session_state.working_df.copy()
            st.session_state.working_df = None
            st.success("Cleaning applied!")
            st.rerun()
        if st.button("⟳ Reset to Original"):
            st.session_state.working_df = original_df.copy()
            st.rerun()

    st.subheader("Current Preview")
    st.dataframe(df.head(10))
    return df

# ──────────────── SIMPLE CHATBOT ────────────────
def nlp_chatbot(question, df):
    if df is None or df.empty:
        return "No data loaded yet."
    try:
        tokens = nltk.word_tokenize(question.lower())
    except Exception as e:
        return f"Tokenization failed: {str(e)}"
    nums = df.select_dtypes(include=np.number).columns.tolist()
    if not nums:
        return "No numeric columns."
    col = nums[0]
    if any(w in tokens for w in ["total", "sum"]):
        return f"**Total {col}**: {df[col].sum():,.2f}"
    if any(w in tokens for w in ["average", "mean", "avg"]):
        return f"**Mean {col}**: {df[col].mean():,.2f}"
    if any(w in tokens for w in ["max", "highest"]):
        return f"**Max {col}**: {df[col].max():,.2f}"
    if "predict" in tokens or "next" in tokens:
        try:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df[col].fillna(0).values
            m = LinearRegression().fit(X, y)
            p = m.predict([[len(df)]])[0]
            return f"**Next predicted {col}**: {p:,.2f} (simple trend)"
        except:
            return "Prediction failed."
    return "Ask about: total, average, max, predict next"

# ──────────────── PDF REPORT ────────────────
def create_report_pdf(text, chart_path=None):
    path = os.path.join(tempfile.gettempdir(), "datagenie_report.pdf")
    doc = SimpleDocTemplate(path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("DataGenie Report", styles["Heading1"]), Spacer(1, 12)]
    for line in text.split("\n"):
        if line.strip():
            elements.append(Paragraph(line, styles["Normal"]))
            elements.append(Spacer(1, 8))
    if chart_path and os.path.exists(chart_path):
        try:
            elements.append(Image(chart_path, width=500, height=350))
        except:
            pass
    doc.build(elements)
    return path

# ──────────────── PAGES ────────────────
def login_page():
    st.title("🔐 DataGenie – Sign In")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        user = login_user(username, password)
        if user:
            st.session_state.update(logged_in=True, user_id=user[0], page="app")
            st.rerun()
        else:
            st.error("Wrong credentials")
    if st.button("Register instead"):
        st.session_state.page = "register"
        st.rerun()

def register_page():
    st.title("📝 Register")
    username = st.text_input("Username")
    pw1 = st.text_input("Password", type="password")
    pw2 = st.text_input("Confirm password", type="password")
    if st.button("Create Account"):
        if pw1 != pw2:
            st.error("Passwords don't match")
        elif len(pw1) < 6:
            st.error("Password too short")
        elif register_user(username, pw1):
            st.success("Account created. Please log in.")
            st.session_state.page = "login"
            st.rerun()
        else:
            st.error("Username taken")
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# ──────────────── MAIN APP ────────────────
def main_app():
    st.title("🪄 DataGenie AI")

    if st.button("Sign Out"):
        for k in list(st.session_state.keys()):
            if k not in ["logged_in", "page"]:
                del st.session_state[k]
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    # ── SIDEBAR ──
    with st.sidebar:
        st.header("📂 Data Controls")
        st.markdown("---")
        uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
        if uploaded is not None and st.session_state.df is None:
            try:
                if uploaded.name.endswith(".csv"):
                    st.session_state.df = pd.read_csv(uploaded)
                else:
                    st.session_state.df = pd.read_excel(uploaded)
                st.session_state.df_filename = uploaded.name
                save_upload(st.session_state.user_id, uploaded.name)
                st.success(f"Loaded: **{uploaded.name}**")
                st.rerun()
            except Exception as e:
                st.error(f"File read error: {e}")

        if st.session_state.df is not None:
            st.markdown("**Current Dataset**")
            st.info(st.session_state.df_filename or "Untitled dataset")
            st.metric("Rows", f"{len(st.session_state.df):,}")
            st.metric("Columns", len(st.session_state.df.columns))
            if st.button("🗑️ Clear Data & Start Over"):
                st.session_state.df = None
                st.session_state.df_filename = None
                st.session_state.working_df = None
                st.rerun()

        st.markdown("---")
        st.caption("Recent uploads")
        for fn, ts in get_uploads(st.session_state.user_id)[:4]:
            st.caption(f"• {fn}")

    # ── EARLY RETURN IF NO DATA ──
    if st.session_state.df is None:
        st.info("Upload a file from the sidebar to begin ✨")
        return

    # From here df is guaranteed to exist
    df = st.session_state.df

    tabs = st.tabs(["🧼 Clean", "📋 Preview", "📈 Charts", "🔍 Insights", "💬 Ask"])

    with tabs[0]:
        updated_df = advanced_cleaning_ui(df)
        # Only update main df when user explicitly applies changes (already handled inside function)

    with tabs[1]:
        st.subheader("Current Data")
        st.dataframe(st.session_state.df)

    with tabs[2]:
        st.subheader("Visualizations")
        x = st.selectbox("X axis", df.columns)
        num_cols = df.select_dtypes(np.number).columns.tolist()
        y = st.selectbox("Y axis", num_cols if num_cols else df.columns, key="y_axis_sel")
        ctype = st.selectbox("Type", ["Bar", "Line", "Pie", "Histogram"])
        fig, ax = plt.subplots()
        try:
            if ctype == "Bar":
                df.groupby(x)[y].sum().plot.bar(ax=ax)
            elif ctype == "Line":
                df.groupby(x)[y].sum().plot.line(ax=ax)
            elif ctype == "Pie":
                df.groupby(x)[y].sum().plot.pie(ax=ax, autopct="%.1f%%")
            elif ctype == "Histogram":
                df[y].plot.hist(ax=ax, bins=30)
            st.pyplot(fig)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            fig.savefig(tmp.name, dpi=120, bbox_inches="tight")
            st.session_state.chart_path = tmp.name
        except Exception as e:
            st.error(f"Chart failed: {e}")

    with tabs[3]:
        st.subheader("Quick Insights")
        nums = df.select_dtypes(np.number).columns[:6]
        lines = []
        for c in nums:
            lines.append(f"**{c}** • total {df[c].sum():,.1f} • avg {df[c].mean():,.2f} • max {df[c].max():,.2f}")
        text = "\n\n".join(lines)
        st.markdown(text)

        if len(nums) > 0:
            try:
                X = np.arange(len(df)).reshape(-1,1)
                y = df[nums[0]].fillna(0).values
                m = LinearRegression().fit(X, y)
                pred = m.predict([[len(df)]])[0]
                st.success(f"Next {nums[0]} ≈ {pred:,.2f}")
                text += f"\n\n**Prediction**: {pred:,.2f}"
            except:
                pass

        if st.button("Download PDF Report"):
            pdf_path = create_report_pdf(text, st.session_state.get("chart_path"))
            with open(pdf_path, "rb") as f:
                st.download_button(
                    label="Download Report",
                    data=f,
                    file_name="datagenie_report.pdf",
                    mime="application/pdf"
                )

    with tabs[4]:
        st.subheader("Chat with Data")
        q = st.text_input("Your question")
        if q:
            ans = nlp_chatbot(q, df)
            st.markdown(f"**Answer:** {ans}")

# ──────────────── ROUTER ────────────────
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    else:
        register_page()
else:
    main_app()
