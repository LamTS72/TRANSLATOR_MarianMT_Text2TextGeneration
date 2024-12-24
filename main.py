import streamlit as st
from transformers import pipeline


st.title('Langchain Demo With Translator Of English to France')
input_text=st.text_input("Input text")

model_path = "Chessmen/translator"
translate_task  = pipeline(
    "translation",
    model=model_path,
)

if input_text:
	st.write(translate_task(input_text))
    