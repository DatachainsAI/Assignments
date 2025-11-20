import streamlit as st
from groq import Groq

client = Groq(api_key='')

BASE_TEMPLATE = """
Rewrite the text in three ways based on the selected role: {role}.

1. Simple version
2. Professional version
3. Translation to {language}

Text: {user_input}

Format:
Simple: <output>
Professional: <output>
Translation: <output>
"""


st.title("📝 Text Enhancer")

user_input = st.text_area("Enter your text")

languages = [
    "English", "French", "Spanish", "German", "Italian", "Portuguese",
    "Chinese (Simplified)", "Chinese (Traditional)", "Japanese", "Korean",
    "Hindi", "Bengali", "Tamil", "Telugu", "Kannada", "Malayalam",
    "Arabic", "Turkish", "Russian", "Ukrainian",
    "Dutch", "Swedish", "Norwegian", "Danish", "Finnish",
    "Polish", "Czech", "Greek", "Thai", "Vietnamese"
]

language = st.selectbox("Translate to", languages, index=1)

temperature = st.slider("Creativity (temperature)", 0.0, 2.0, 0.7)

role = st.selectbox(
    "Role",
    ["Teacher", "Analyst", "Storyteller"]
)

if st.button("Generate"):
    if not user_input.strip():
        st.warning("Please enter some text.")
    else:
        prompt = BASE_TEMPLATE.format(
            role=role,
            language=language,
            user_input=user_input
        )
        print(prompt)

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )

        output = response.choices[0].message.content

        st.subheader("✨ Output")
        st.write(output)
