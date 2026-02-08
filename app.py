import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- HEADER ----------
st.markdown("""
    <h1 style='text-align: center;'>🤖 DataGenie</h1>
    <p style='text-align: center; color: gray;'>AI‑Powered Decision Support Dashboard</p>
""", unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset", type=["xlsx"])

# ---- Demo login system ----
st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Demo Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_btn = st.sidebar.button("Login")

logged_in = False
if login_btn:
    if username == "admin" and password == "1234":
        st.sidebar.success("Login successful")
        logged_in = True
    else:
        st.sidebar.error("Invalid credentials")

ai_text = ""
df = None

# ---------- AFTER UPLOAD ----------
if uploaded_file is not None and logged_in:
    df = pd.read_excel(uploaded_file)

    # ---------- CLEANING ----------
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    # ---------- KPI CARDS ----------
    if "Sales" in df.columns:
        total_sales = df["Sales"].sum()
        avg_sales = df["Sales"].mean()
        max_sales = df["Sales"].max()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Sales", f"{total_sales:,.0f}")
        col2.metric("Average Sales", f"{avg_sales:,.0f}")
        col3.metric("Highest Sale", f"{max_sales:,.0f}")

    st.divider()

    # ---------- TABS ----------
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 Data", "📊 Dashboard", "🤖 AI Insights", "💬 Chatbot"
    ])

    # ---------- TAB 1: DATA ----------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df, use_container_width=True)

    # ---------- TAB 2: DASHBOARD ----------
    with tab2:
        st.subheader("📊 Advanced Interactive Dashboard")

        if "Sales" in df.columns:

            colA, colB = st.columns(2)

            # Filters
            if "Region" in df.columns:
                regions = colA.multiselect("Region", df["Region"].unique(), default=df["Region"].unique())
                df_filtered = df[df["Region"].isin(regions)]
            else:
                df_filtered = df.copy()

            if "Category" in df.columns:
                categories = colB.multiselect("Category", df_filtered["Category"].unique(), default=df_filtered["Category"].unique())
                df_filtered = df_filtered[df_filtered["Category"].isin(categories)]

            # Bar chart
            if "Category" in df_filtered.columns:
                st.markdown("### Sales by Category")
                cat_sales = df_filtered.groupby("Category")["Sales"].sum()
                fig1, ax1 = plt.subplots()
                cat_sales.plot(kind="bar", ax=ax1)
                st.pyplot(fig1)

            # Pie chart
            if "Region" in df_filtered.columns:
                st.markdown("### Region Distribution")
                region_sales = df_filtered.groupby("Region")["Sales"].sum()
                fig2, ax2 = plt.subplots()
                region_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
                ax2.set_ylabel("")
                st.pyplot(fig2)

            # Line trend
            st.markdown("### Sales Trend Over Time")
            trend = df_filtered["Sales"].reset_index(drop=True)
            fig3, ax3 = plt.subplots()
            ax3.plot(trend)
            ax3.set_xlabel("Order Index")
            ax3.set_ylabel("Sales")
            st.pyplot(fig3)

        else:
            st.warning("Sales column not found in dataset.")

    # ---------- TAB 3: AI INSIGHTS ----------
    with tab3:
        st.subheader("🤖 AI Insights & Forecast")

        if "Sales" in df.columns:
            total_sales = df["Sales"].sum()
            avg_sales = df["Sales"].mean()
            max_sales = df["Sales"].max()

            top_region = "N/A"
            if "Region" in df.columns:
                top_region = df.groupby("Region")["Sales"].sum().idxmax()

            ai_text = f"""
Total Sales: {total_sales:,.2f}

Average Sales per Order: {avg_sales:,.2f}

Highest Single Sale: {max_sales:,.2f}

Top Performing Region: {top_region}

AI Recommendation:
Focus marketing and inventory in the {top_region} region to maximize revenue growth.
"""

            st.info(ai_text)

            # -------- FUTURE FORECAST (10 steps) --------
            st.subheader("📈 Future Sales Forecast (Next 10)")

            X = np.arange(len(df)).reshape(-1, 1)
            y = df["Sales"].values

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(df), len(df) + 10).reshape(-1, 1)
            future_preds = model.predict(future_X)

            fig4, ax4 = plt.subplots()
            ax4.plot(range(len(df)), y)
            ax4.plot(range(len(df), len(df) + 10), future_preds)
            ax4.set_xlabel("Order Index")
            ax4.set_ylabel("Sales")
            st.pyplot(fig4)

            # -------- PDF --------
            def create_pdf(text):
                file_path = "/tmp/datagenie_report.pdf"
                doc = SimpleDocTemplate(file_path)
                styles = getSampleStyleSheet()
                elements = []

                for line in text.split("\n"):
                    elements.append(Paragraph(line, styles["Normal"]))
                    elements.append(Spacer(1, 12))

                doc.build(elements)
                return file_path

            if st.button("📄 Download AI Report"):
                pdf_path = create_pdf(ai_text)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, "DataGenie_Report.pdf")

    # ---------- TAB 4: CHATBOT ----------
    with tab4:
        st.subheader("💬 AI‑Style Data Chatbot")

        question = st.text_input("Ask anything about your data...")

        if question and "Sales" in df.columns:
            q = question.lower()

            if "total" in q and "sales" in q:
                st.success(f"Total sales is {df['Sales'].sum():,.2f}.")

            elif "average" in q:
                st.success(f"Average sales per order is {df['Sales'].mean():,.2f}.")

            elif "highest" in q and "region" in q and "Region" in df.columns:
                region = df.groupby("Region")["Sales"].sum().idxmax()
                st.success(f"Highest performing region is {region}.")

            elif "predict" in q or "forecast" in q:
                X = np.arange(len(df)).reshape(-1, 1)
                y = df["Sales"].values
                model = LinearRegression()
                model.fit(X, y)
                next_val = model.predict([[len(df)]])[0]
                st.success(f"Next predicted sales value is {next_val:,.2f}.")

            else:
                st.info("Try asking about total sales, average sales, top region, or future forecast.")

else:
    st.info("⬅️ Login and upload an Excel dataset to use DataGenie.")
