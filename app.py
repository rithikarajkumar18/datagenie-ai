import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from openai import OpenAI
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="DataGenie AI", layout="wide")

st.title("🤖 DataGenie – AI Powered Decision Support System")

# OpenAI client
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

ai_text = ""

if uploaded_file:
    df = pd.read_excel(uploaded_file)

    st.subheader("📄 Raw Data")
    st.dataframe(df)

    # -------- DATA CLEANING --------
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

    st.success("✅ Data cleaned successfully!")

    # -------- CHART --------
    if "Category" in df.columns and "Sales" in df.columns:
        st.subheader("📊 Sales by Category")
        cat_sales = df.groupby("Category")["Sales"].sum()

        fig, ax = plt.subplots()
        cat_sales.plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # -------- AI INSIGHTS --------
    if "Sales" in df.columns:
        st.subheader("🤖 AI Generated Insights")

        summary = df.describe().to_string()

        prompt = f"""
        You are a business analyst.
        Analyze this sales summary and give insights and recommendations:

        {summary}
        """

        st.subheader("🤖 AI Generated Insights (Demo Mode)")

ai_text = ""

if "Sales" in df.columns:
    total_sales = df["Sales"].sum()
    avg_sales = df["Sales"].mean()
    max_sales = df["Sales"].max()

    top_region = "N/A"
    if "Region" in df.columns:
        top_region = df.groupby("Region")["Sales"].sum().idxmax()

    ai_text = f"""
    • Total Sales: {total_sales:,.2f}

    • Average Sales per Order: {avg_sales:,.2f}

    • Highest Single Sale: {max_sales:,.2f}

    • Top Performing Region: {top_region}

    📊 Insight:
    Sales performance is strong in the {top_region} region.
    Focus on increasing marketing and inventory in this region
    to maximize revenue. Monitor low-sales regions for improvement.
    """

    st.write(ai_text)


    # -------- PDF DOWNLOAD --------
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

    if ai_text and st.button("📄 Download AI Report as PDF"):
        pdf_path = create_pdf(ai_text)
        with open(pdf_path, "rb") as f:
            st.download_button("⬇️ Download PDF", f, file_name="DataGenie_Report.pdf")
