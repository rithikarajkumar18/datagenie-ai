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

ai_text = ""
df = None

# ---------- AFTER UPLOAD ----------
if uploaded_file is not None:
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
        st.subheader("Sales by Category")

        if "Category" in df.columns and "Sales" in df.columns:
            cat_sales = df.groupby("Category")["Sales"].sum()

            fig, ax = plt.subplots()
            cat_sales.plot(kind="bar", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Required columns not found.")

    # ---------- TAB 3: AI INSIGHTS ----------
    with tab3:
        st.subheader("AI Generated Insights")

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

Insight:
Focus marketing and inventory in the {top_region} region to increase revenue.
"""

            st.info(ai_text)

            # -------- ML SALES PREDICTION --------
            st.subheader("📈 Next Sale Prediction")

            X = np.arange(len(df)).reshape(-1, 1)
            y = df["Sales"].values

            model = LinearRegression()
            model.fit(X, y)

            next_sales = model.predict([[len(df)]])[0]
            st.success(f"Predicted next sale value: {next_sales:,.2f}")

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
        st.subheader("💬 Smart Data Chatbot")

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

            elif "predict" in q or "next" in q:
                X = np.arange(len(df)).reshape(-1, 1)
                y = df["Sales"].values

                model = LinearRegression()
                model.fit(X, y)
                next_sales = model.predict([[len(df)]])[0]

                st.success(f"Predicted next sales value is {next_sales:,.2f}.")

            else:
                st.info("Try asking about total, average, region performance, or prediction.")

        else:
            st.info("⬅️ Upload an Excel file from the sidebar to begin.")
