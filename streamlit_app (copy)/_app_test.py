import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Add the parent directory to the Python path
import pandas as pd
import streamlit as st
from openai import OpenAI

from utils import convert_pdf_to_images
from paths import OUTPUT_IMAGES_PATH , INPUT_PDFS_PATH, SURYA_TEXT_OUTPUT_FILE, SURYA_JSON_TEMP
from config import load_config
from openai_local.refrence_dictionaries.target_fields import expropriation_data
from openai_local.utils import get_field_from_text, setup_openai
from openai_local.prompts import MAIN_PROMPT
from app_constants import STAGES
from surya_ocr_utils import from_surya_to_txt

# Load configuration
config = load_config()

# Set page configuration
st.set_page_config(page_title="RCAR E-Consignation PoC Local", page_icon=":keyboard:")

# App title
st.title("RCAR E-Consignation PoC Local")

# File upload control
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf", key="unique_pdf_uploader")

if uploaded_file is not None:
    # Validate file type
    if uploaded_file.type != "application/pdf":
        st.error("Please upload a valid PDF file.")
    else:
        client = OpenAI(api_key=setup_openai())

        # Start the process
        st.success("File uploaded successfully! Processing will start now.")

        # Progress bar initialization
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Step 1: Save uploaded file locally and convert PDF to images
        status_text.text(STAGES[0])
        pdf_file_name = uploaded_file.name
        pdf_file_path = os.path.join(INPUT_PDFS_PATH, pdf_file_name)

        with open(pdf_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())  # Save uploaded file locally

        image_files = convert_pdf_to_images(pdf_file_name, INPUT_PDFS_PATH, OUTPUT_IMAGES_PATH)
        progress_bar.progress(1 / len(STAGES))

        # Step 7: Extract text
        status_text.text(STAGES[6])
        extracted_text = from_surya_to_txt(OUTPUT_IMAGES_PATH, SURYA_TEXT_OUTPUT_FILE, SURYA_JSON_TEMP)
        progress_bar.progress(7 / len(STAGES))

        # Step 9: Extract fields using OpenAI
        status_text.text(STAGES[8])

        if "feedback_df" not in st.session_state:
            st.session_state["feedback_df"] = pd.DataFrame(columns=[
                "file_path", "field_name", "LLM answer", "True", "False", "User Answer"
            ])
        if "field_index" not in st.session_state:
            st.session_state["field_index"] = 0

        field_index = st.session_state["field_index"]

        if field_index < len(expropriation_data):
            # Process one field at a time
            item = expropriation_data[field_index]
            field = item.get("field")
            field_prompt = item.get("field_prompt")
            lookuptext = item.get("lookuptext")
            dynamic = item.get("dynamic")
            default_value = item.get("default_value")

            # Use OpenAI to extract field or set to default value
            if dynamic:
                response = get_field_from_text(client, MAIN_PROMPT, field_prompt, extracted_text, field)
            else:
                response = default_value

            # Display current field for user feedback
            st.write(f"### Field {field_index + 1}/{len(expropriation_data)}: {field}")
            st.write(f"**LLM Answer**: {response}")

            # Collect feedback
            with st.form(f"form_{field_index}"):
                is_true = st.checkbox("True", key=f"true_{field_index}")
                is_false = st.checkbox("False", key=f"false_{field_index}")
                user_answer = st.text_input("Your Answer:", key=f"user_answer_{field_index}")

                submitted = st.form_submit_button("Submit Feedback")
                if submitted:
                    st.session_state["feedback_df"] = st.session_state["feedback_df"]._append({
                        "file_path": pdf_file_name,
                        "field_name": field,
                        "LLM answer": response,
                        "True": is_true,
                        "False": is_false,
                        "User Answer": user_answer
                    }, ignore_index=True)
                    st.session_state["field_index"] += 1
                    # Update query parameters to manage state progression
                    st.session_state.query_params = st.session_state.get("field_index", 0)

        else:
            # Output the final feedback DataFrame
            st.success("Feedback collection completed!")
            st.write("### Final Feedback DataFrame:")
            st.dataframe(st.session_state["feedback_df"])

            # Download button for feedback DataFrame
            csv_output = st.session_state["feedback_df"].to_csv(index=False)
            st.download_button(
                label="Download Feedback CSV",
                data=csv_output,
                file_name="feedback_results.csv",
                mime="text/csv"
            )

            progress_bar.progress(10 / len(STAGES))

else:
    st.info("Please upload a PDF document to begin.")
