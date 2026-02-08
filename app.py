import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="DataGenie AI", layout="wide")

# ---------- SESSION STATE ----------
if "users" not in st.session_state:
    st.session_state.users = {"admin": "1234"}

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

# ---------- LOGIN PAGE ----------
def login_page():
    st.markdown("""
        <h1 style='text-align: center;'>🤖 DataGenie</h1>
        <p style='text-align: center; color: gray;'>AI-Powered Decision Support Dashboard</p>
    """, unsafe_allow_html=True)

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in st.session_state.users and st.session_state.users[username] == password:
            st.session_state.logged_in = True
            st.session_state.page = "app"
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown("---")
    if st.button("Create new account"):
        st.session_state.page = "register"
        st.rerun()


# ---------- REGISTER PAGE ----------
def register_page():
    st.markdown("""
        <h1 style='text-align: center;'>📝 Register for DataGenie</h1>
    """, unsafe_allow_html=True)

    new_user = st.text_input("New Username")
    new_pass = st.text_input("New Password", type="password")
    confirm_pass = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if not new_user or not new_pass:
            st.warning("Please fill all fields")
        elif new_user in st.session_state.users:
            st.error("Username already exists")
        elif new_pass != confirm_pass:
            st.error("Passwords do not match")
        else:
            st.session_state.users[new_user] = new_pass
            st.success("Account created successfully! Please login.")

    st.markdown("---")
    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


# ---------- MAIN APP ----------
def main_app():
    # ---------- HEADER ----------
    col_left, col_right = st.columns([9, 1])  # 9:1 ratio, left for title, right for button
    with col_left:
        st.markdown("""
            <h1 style='text-align: left;'>🤖 DataGenie</h1>
            <p style='text-align: left; color: gray;'>AI-Powered Decision Support Dashboard</p>
    """, unsafe_allow_html=True)

    with col_right:
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    # ---------- SIDEBAR ----------
    st.sidebar.title("⚙️ Controls")
    uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["xlsx", "csv"])

    ai_text = ""
    df = None

    # ---------- AFTER UPLOAD ----------
    if uploaded_file is not None:
       if uploaded_file.name.endswith(".csv"):
           df = pd.read_csv(uploaded_file)
       else:
           df = pd.read_excel(uploaded_file)

    # ---------- CLEANING ----------
    def clean_data(df):
        df = df.dropna()
        df = df.drop_duplicates() 
        return df

    if df is not None:
        df = clean_data(df)
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

        # ---------- TAB 1 ----------
        with tab1:
            st.subheader("Dataset Preview")
            st.dataframe(df, use_container_width=True)

        # ---------- TAB 2 ----------
        with tab2:
            st.subheader("📊 Interactive Sales Dashboard")

            if "Sales" in df.columns:
                colA, colB = st.columns(2)

                if "Region" in df.columns:
                    regions = colA.multiselect("Filter by Region", df["Region"].unique(), default=df["Region"].unique())
                    df_filtered = df[df["Region"].isin(regions)]
                else:
                    df_filtered = df.copy()

                if "Category" in df.columns:
                    categories = colB.multiselect("Filter by Category", df_filtered["Category"].unique(), default=df_filtered["Category"].unique())
                    df_filtered = df_filtered[df_filtered["Category"].isin(categories)]

                if "Category" in df_filtered.columns:
                    st.markdown("### Sales by Category")
                    cat_sales = df_filtered.groupby("Category")["Sales"].sum()

                    fig1, ax1 = plt.subplots()
                    cat_sales.plot(kind="bar", ax=ax1)
                    st.pyplot(fig1)

                if "Region" in df_filtered.columns:
                    st.markdown("### Sales Distribution by Region")
                    region_sales = df_filtered.groupby("Region")["Sales"].sum()

                    fig2, ax2 = plt.subplots()
                    region_sales.plot(kind="pie", autopct="%1.1f%%", ax=ax2)
                    ax2.set_ylabel("")
                    st.pyplot(fig2)

                st.markdown("### Sales Trend")
                trend = df_filtered["Sales"].reset_index(drop=True)

                fig3, ax3 = plt.subplots()
                ax3.plot(trend)
                ax3.set_xlabel("Order Index")
                ax3.set_ylabel("Sales")
                st.pyplot(fig3)

            else:
                st.warning("Sales column not found in dataset.")

        # ---------- TAB 3 ----------
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

                st.subheader("📈 Next Sale Prediction")

                X = np.arange(len(df)).reshape(-1, 1)
                y = df["Sales"].values

                model = LinearRegression()
                model.fit(X, y)

                next_sales = model.predict([[len(df)]])[0]
                st.success(f"Predicted next sale value: {next_sales:,.2f}")

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

        # ---------- TAB 4 ----------
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


# ---------- ROUTER ----------
if not st.session_state.logged_in:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "register":
        register_page()
else:
    main_app()
