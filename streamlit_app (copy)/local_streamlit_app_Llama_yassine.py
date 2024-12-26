import sys
import os
# from streamlit_app.surya_ocr_utils import from_surya_to_txt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add the parent directory to the Python path
import streamlit as st
from datetime import datetime
import pandas as pd
import torch
from paths import OUTPUT_IMAGES_PATH, INPUT_PDFS_PATH, SURYA_TEXT_OUTPUT_FILE, SURYA_JSON_TEMP , TEMP_DIR_ABS
from config import load_config
from openai_local.refrence_dictionaries.target_fields import expropriation_data
from utils import extract_pdf_content , get_qa_chain , convert_pdf_to_images , process_csvs_to_dataframe, execute_safe_command, process_csvs_to_excel,get_names_arabert,add_beneficiaries
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For text chunking
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings
from langchain_community.vectorstores import FAISS  # Vector database
from surya_ocr_utils import from_surya_to_txt

from constants import STAGES

# Configuration Tesseract pour l'OCR arabe
os.environ['TESSDATA_PREFIX'] = '/usr/share/tesseract-ocr/4.00/tessdata'


# Load configuration
config = load_config()

# Set page configuration
st.set_page_config(page_title="RCAR E-Consignation Llama Local", page_icon=":keyboard:")

# App title
st.title("RCAR E-Consignation PoC Local")

# File upload control for multiple PDFs
uploaded_files = st.file_uploader(
    "Upload PDF documents", type="pdf", accept_multiple_files=True, key="unique_pdf_uploader"
)

# Add a button to start processing
if uploaded_files:
    invalid_files = [file.name for file in uploaded_files if file.type != "application/pdf"]
    if invalid_files:
        st.error(f"Invalid files detected (not PDFs): {', '.join(invalid_files)}. Please upload only PDF files.")
    else:
        if st.button("Start Processing"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_files = len(uploaded_files)

            # Iterate over each uploaded PDF
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(STAGES[0])  # Updating stage text
                pdf_file_name = uploaded_file.name
                pdf_file_path = os.path.join(INPUT_PDFS_PATH, pdf_file_name)

                # Save uploaded PDF locally
                with open(pdf_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                st.info(f"Processing {pdf_file_name}...")

                # Step 2: Convert PDF to images
                status_text.text(STAGES[1])
                # transform pdf into images

                # extracted_text = extract_pdf_content(pdf_file_path)
                # print(extracted_text)

                image_files = convert_pdf_to_images(pdf_file_path, INPUT_PDFS_PATH, OUTPUT_IMAGES_PATH)
                # Extract txt from images using surya ocr
                extracted_texts = []
                for image_file in image_files:
                    extracted_text = from_surya_to_txt(image_file,SURYA_TEXT_OUTPUT_FILE, SURYA_JSON_TEMP)
                    extracted_texts.append(extracted_text)
                extracted_text = '\n'.join(extracted_texts) # concatenated text of all extracted texts from images
                directory = os.path.join(TEMP_DIR_ABS, pdf_file_name)
                # print(directory)

                # Get the file name without the extension
                CSV_DIR = os.path.splitext(directory)[0]

                # todo: add safe command execution to avoid injections
                # Construct the tabled command with the required format and pass the image file
                tabled_command = f"tabled '{pdf_file_path}' temp_html_output --format csv"


                # Execute the tabled command as a shell command
                os.system(tabled_command)


                # Step 3: Process one image at a time
                status_text.text(STAGES[2])
                feedback_df = pd.DataFrame(columns=["field_name", "LLM answer", "Correct Answer"])

                # Initialize all rows in feedback_df with "NA"
                for item in expropriation_data:
                    feedback_df = feedback_df._append(
                        {"field_name": item.get("field"), "LLM answer": "NA", "Correct Answer": "NA"},
                        ignore_index=True)

                # Segmentation du contenu
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=50,  # 10% Chunk
                )
                text_chunks = text_splitter.split_text(extracted_text)

                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L12-v2"
                )

                VECTOR_STORE = FAISS.from_texts(text_chunks, embeddings)

                # Configuration de la chaîne de question-réponse
                qa_chain = get_qa_chain(VECTOR_STORE)

                # Ask questions for fields with "NA" in the DataFrame
                for row_index in range(len(feedback_df)):
                    field = expropriation_data[row_index].get("field")
                    field_prompt = expropriation_data[row_index].get("field_prompt")
                    default_value = expropriation_data[row_index].get("default_value")
                    # Pass only the text from the current image to the LLM
                    response = qa_chain.invoke({"query": field_prompt})["result"]
                    # Update the DataFrame with the response
                    feedback_df.at[row_index, "LLM answer"] = response


                # Step 5: Export data to Excel
                status_text.text(STAGES[4])
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file_name = f"{os.path.splitext(pdf_file_name)[0]}_{timestamp}_output.xlsx"
                output_file_path = os.path.join(INPUT_PDFS_PATH, output_file_name)
                feedback_df.to_excel(output_file_path, index=False)

                # Step 6: Loop over all HTMLs and output an Excel of concatenated tables
                # concat_tables = process_csvs_to_dataframe(CSV_DIR, feedback_df,output_file_path)
                # concat_tables.to_excel(output_file_path)
                code_status = process_csvs_to_excel(CSV_DIR,output_file_path)
                if code_status:
                    people_names = get_names_arabert(output_file_path)
                    add_beneficiaries(people_names,output_file_path)
                    print(people_names)
                    st.write(people_names)
                # Print the head of the concatenated tables as required
                # print(concat_tables.head(20))

                # Step 6: Clean up temporary files
                status_text.text(STAGES[5])
                for file in os.listdir(OUTPUT_IMAGES_PATH):
                    os.remove(os.path.join(OUTPUT_IMAGES_PATH, file))

                progress_bar.progress((i + 1) / total_files)
                st.success(f"Processed {pdf_file_name}. Output saved as {output_file_name}.")
                torch.cuda.empty_cache() # empty gpu cache memory
            status_text.text(STAGES[6])

            st.success("All files processed successfully!")
else:
    st.info("Please upload one or more PDF documents to begin.")
