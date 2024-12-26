import os

# Define the base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR,'data')
STREAMLIT_ASSETS = os.path.join(BASE_DIR,'streamlit_app/Assets')

# Input and output paths
INPUT_PDFS_PATH = os.path.join(DATA_DIR, 'input_pdfs')
OUTPUT_IMAGES_PATH = os.path.join(DATA_DIR, "output_images/" )
OUTPUT_TEXT_PATH = os.path.join(DATA_DIR, 'extracted_text')
SURYA_TEXT_OUTPUT_FILE=os.path.join(OUTPUT_TEXT_PATH,'surya_temp_json.txt')
PROCESSED_IMGES_PATH = os.path.join(DATA_DIR,'processed_images')
DENOISED_PATH = os.path.join(DATA_DIR,'denoised_images')
SURYA_JSON_TEMP = os.path.join(DATA_DIR,"json_output_temp.json")
TEMP_DIR = r'./temp_html_output'
TEMP_DIR_ABS = os.path.abspath(TEMP_DIR)
