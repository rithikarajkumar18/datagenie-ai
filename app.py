import streamlit as st
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