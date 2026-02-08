import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="DataGenie AI", layout="wide")

st.title("🤖 DataGenie – AI Powered Decision Support System")

uploaded_file = st.file_uploader("Upload Excel Dataset", type=["xlsx"])

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

    # -------- FILTERS --------
    st.sidebar.header("🔎 Filters")

    if "Region" in df.columns:
        region = st.sidebar.selectbox("Select Region", df["Region"].unique())
        df = df[df["Region"] == region]

    if "Category" in df.columns:
        category = st.sidebar.selectbox("Select Category", df["Category"].unique())
        df = df[df["Category"] == category]

    # -------- CHART 1 --------
    if "Category" in df.columns and "Sales" in df.columns:
        st.subheader("📊 Sales by Category")
        cat_sales = df.groupby("Category")["Sales"].sum()

        fig1, ax1 = plt.subplots()
        cat_sales.plot(kind="bar", ax=ax1)
        st.pyplot(fig1)

    # -------- CHART 2 --------
    if "Region" in df.columns and "Sales" in df.columns:
        st.subheader("🌍 Sales by Region")
        region_sales = df.groupby("Region")["Sales"].sum()

        fig2, ax2 = plt.subplots()
        region_sales.plot(kind="bar", ax=ax2)
        st.pyplot(fig2)

    # -------- CHART 3 --------
    if "Date" in df.columns and "Sales" in df.columns:
        st.subheader("📅 Monthly Sales Trend")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        monthly = df.groupby(df["Date"].dt.month)["Sales"].sum()

        fig3, ax3 = plt.subplots()
        monthly.plot(ax=ax3)
        st.pyplot(fig3)

    # -------- AI INSIGHTS --------
    if "Sales" in df.columns:
        st.subheader("🤖 AI Decision Insights")

        avg_sales = df["Sales"].mean()
        max_sales = df["Sales"].max()
        min_sales = df["Sales"].min()

        st.write(f"🔹 Average Sales: ₹{avg_sales:,.2f}")
        st.write(f"🔹 Maximum Sales: ₹{max_sales:,.2f}")
        st.write(f"🔹 Minimum Sales: ₹{min_sales:,.2f}")

        st.markdown("### 📌 Business Recommendation")
        st.info(
            "Increase focus on high-performing regions and improve marketing strategies in low-sales areas to maximize revenue growth."
        )

    # -------- SUMMARY --------
    st.subheader("📈 Summary Statistics")
    st.write(df.describe())
