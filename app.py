import os
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "Sales" in df.columns:
    st.subheader("🤖 AI Generated Business Insights")

    data_summary = df.describe().to_string()

    prompt = f"""
    You are a business analyst.
    Analyze the following sales data summary and give clear insights and recommendations:

    {data_summary}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    ai_text = response.choices[0].message.content
    st.write(ai_text)
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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
        st.download_button("⬇️ Click to Download PDF", f, file_name="DataGenie_Report.pdf")

