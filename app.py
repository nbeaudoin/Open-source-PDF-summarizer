# ChatGPT used to help build this

import streamlit as st
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pycryptodome

# Function to read and extract text from the uploaded PDF using PdfReader
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to summarize text using LLaMA model or similar
def summarize_text(text, tokenizer, model, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], max_length=max_length, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Load the LLaMA model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "facebook/llama-7b"  # Replace this with actual LLaMA model if available on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

# Streamlit app UI
def main():
    st.title("PDF Summarizer using LLaMA Model")

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        st.info("Extracting text from PDF...")
        text = extract_text_from_pdf(uploaded_file)
        st.text_area("Extracted Text", text[:2000], height=300)  # Preview of extracted text

        if st.button("Summarize"):
            tokenizer, model = load_model()
            st.info("Summarizing the extracted text...")
            summary = summarize_text(text, tokenizer, model)
            st.success("Summary:")
            st.write(summary)

if __name__ == "__main__":
    main()



