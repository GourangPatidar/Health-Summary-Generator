import streamlit as st
import pdfplumber
from transformers import pipeline

# Initialize the summarization pipeline
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="facebook/bart-large-cnn")

summarizer = load_summarizer()

# Streamlit app interface
st.title("Health Summary Generator Report")
st.write("Upload a PDF file and get a Report")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    try:
        # Extract text from the PDF using PDFplumber
        with pdfplumber.open(uploaded_file) as pdf:
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += text + "\n"
        
        if all_text.strip():
            st.subheader("Extracted Text")
            st.text_area("Extracted Text from PDF", all_text, height=300)

            # Summarize the text
            st.subheader("Summarized Text")
            if st.button("Report"):
                with st.spinner("Just done..."):
                    # Break text into smaller chunks for summarization
                    chunk_size = 1024
                    chunks = [all_text[i:i+chunk_size] for i in range(0, len(all_text), chunk_size)]
                    summary = ""
                    for chunk in chunks:
                        summarized_chunk = summarizer(chunk, max_length=150, min_length=50, do_sample=False)
                        summary += summarized_chunk[0]['summary_text'] + " "
                    
                    st.success("Report complete!")
                    st.text_area("Summary", summary.strip(), height=200)
        else:
            st.error("No text found in the PDF. Ensure it's not a scanned image-based PDF.")
    
    except Exception as e:
        st.error(f"An error occurred while processing the PDF: {e}")

else:
    st.info("Please upload a PDF file to begin.")

# Footer
st.write("---")
st.write("developed by GOURANG PATIDAR ")
st.write("Developed using [Streamlit](https://streamlit.io/) and [Hugging Face Transformers](https://huggingface.co/transformers).")
