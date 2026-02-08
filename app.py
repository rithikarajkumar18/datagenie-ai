import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="DataGenie AI", layout="wide")

st.title("🤖 DataGenie – AI Powered Decision Support System")

# ---------- SIDEBAR ----------
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload Excel Dataset", type=["xlsx"])

ai_text = ""
df = None

# ---------- AFTER UPLOAD ----------
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 Data", "📊 Dashboard", "🤖 AI Insights", "💬 Chatbot"]
    )

    # ---------- TAB 1: DATA ----------
    with tab1:
        st.subheader("Raw Data")
        st.dataframe(df)

    # ---------- CLEANING ----------
    df = df.drop_duplicates()

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].mean())

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("Unknown")

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
        st.subheader("AI Generated Insights (Demo Mode)")

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
            st.write(ai_text)

            # PDF
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

            if st.button("📄 Download AI Report as PDF"):
                pdf_path = create_pdf(ai_text)
                with open(pdf_path, "rb") as f:
                    st.download_button("⬇️ Download PDF", f, "DataGenie_Report.pdf")

    # ---------- TAB 4: CHATBOT ----------
    with tab4:
        st.subheader("Ask Questions About Your Data")

        question = st.text_input("Type your question here...")

        if question and "Sales" in df.columns:
            if "highest" in question.lower() and "region" in question.lower():
                answer = df.groupby("Region")["Sales"].sum().idxmax()
                st.success(f"Highest sales region is: {answer}")

            elif "total sales" in question.lower():
                st.success(f"Total sales is: {df['Sales'].sum():,.2f}")

            elif "average sales" in question.lower():
                st.success(f"Average sales is: {df['Sales'].mean():,.2f}")

            else:
                st.info("Demo chatbot: Try asking about total, average, or highest sales region.")
