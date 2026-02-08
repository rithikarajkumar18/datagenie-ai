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
