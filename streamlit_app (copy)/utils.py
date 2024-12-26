
import pandas as pd

import shlex
import subprocess
import os
import shutil
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from datetime import datetime
import io
import cv2
import numpy as np

from openai_local.prompts import MAIN_PROMPT
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TesseractOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_ollama.llms import OllamaLLM  # Local LLM integration
from langchain.text_splitter import RecursiveCharacterTextSplitter  # For text chunking
from langchain_huggingface import HuggingFaceEmbeddings  # For text embeddings
from langchain_community.vectorstores import FAISS  # Vector database
from langchain.chains import RetrievalQA  # For question-answering pipeline
from langchain.prompts import PromptTemplate  # For customizing LLM prompts
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline


# Ensure the command does not contain multiple commands
def execute_safe_command(command):
    # Split the command into parts to validate
    parts = shlex.split(command)

    if len(parts) == 0:
        raise ValueError("Command cannot be empty.")

    # Simple validation to ensure no shell metacharacters for chaining
    if any(char in command for char in [';', '&&', '||', '|']):
        raise ValueError("Multiple commands or shell operators are not allowed.")

    # Execute the command safely
    result = subprocess.run(parts, check=True, text=True, capture_output=True)
    return result.stdout

def process_html_to_dataframe(folder_path, output_excel_path):
    """
    Processes all HTML files in a given folder, converts them to DataFrames, and concatenates them.
    Exports the concatenated DataFrame to an Excel file with Arabic support.

    Parameters:
        folder_path (str): The path to the folder containing HTML files.
        output_excel_path (str): The path to save the Excel file.

    Returns:
        pd.DataFrame: A concatenated DataFrame of all HTML files, or an empty DataFrame if no HTMLs are found.
    """
    # List to store DataFrames
    dataframes = []

    # Loop over files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.html'):
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read HTML file into a list of DataFrames
                html_tables = pd.read_html(file_path)
                # Add each table DataFrame to the list
                dataframes.extend(html_tables)
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    # Concatenate DataFrames if any are found
    if dataframes:
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        # Export to Excel

        concatenated_df.to_csv(output_excel_path, index=False)
        return concatenated_df
    else:
        # Create an empty DataFrame and export to Excel
        empty_df = pd.DataFrame()
        empty_df.to_csv(output_excel_path, index=False)
        return empty_df

def process_csvs_to_dataframe(folder_path, feedback_df, output_file = ''):
    """
    Processes all CSV files in a given folder, converts them to DataFrames, and adds each as a separate sheet to feedback_df.
    The first created CSV file's headers are used as column names for all DataFrames.
    Exports the feedback_df to an Excel file.

    Parameters:
        folder_path (str): The path to the folder containing CSV files.
        feedback_df (pd.ExcelWriter): An ExcelWriter object to store the DataFrames as separate sheets.
        output_excel_path (str): The path to save the Excel file.

    Returns:
        pd.ExcelWriter: The feedback_df with all CSVs added as separate sheets.
    """
    # Get list of CSV files sorted by creation time
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    csv_files.sort(key=lambda f: os.path.getctime(os.path.join(folder_path, f)))

    if not csv_files:
        # No CSV files found
        print("No CSV files found in the folder.")
        return feedback_df

    # Process and add each CSV as a separate sheet
    for idx, file_name in enumerate(csv_files):
        file_path = os.path.join(folder_path, file_name)
        try:
            # Read the CSV with headers
            df = pd.read_csv(file_path)
            print(df)
            sheet_name = f"Sheet{idx + 1}"
            df.to_excel(feedback_df, sheet_name=sheet_name, index=False, engine='openpyxl')
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    return feedback_df
def process_csvs_to_excel(folder_path, output_excel_path):
    """
    Processes all CSV files in a given folder, converts them to DataFrames, and adds each as a separate sheet to an Excel file.
    The first created CSV file's headers are used as column names for all DataFrames.
    Exports the final Excel file.

    Parameters:
        folder_path (str): The path to the folder containing CSV files.
        output_excel_path (str): The path to save the Excel file.

    Returns:
        None
    """
    try:
    # Get list of CSV files sorted by creation time
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        csv_files.sort(key=lambda f: os.path.getctime(os.path.join(folder_path, f)))

        if not csv_files:
            print("No CSV files found in the folder.")
            return

        # Create an ExcelWriter object
        with pd.ExcelWriter(output_excel_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
            for idx, file_name in enumerate(csv_files):
                file_path = os.path.join(folder_path, file_name)
                print(f"Processing file: {file_path}")
                try:
                    # Read the CSV with headers
                    df = pd.read_csv(file_path)
                    sheet_name = f"Sheet{idx + 2}"
                    # Write DataFrame to a new sheet in the Excel file
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                except Exception as e:
                    print(f"Error reading {file_name}: {e}")

        print(f"Excel file created at: {output_excel_path}")
        return True
    except Exception as e:
        print(f"Error creating csv: {e}")
        return False

def add_beneficiaries(list_names,file_path):
    df = pd.DataFrame(list_names, columns=['bénéficiaires'])
    sheet_name = "bénéficiaires"
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    print("Beneficiaries written in the excel file")
def process_excel_to_single_column(file_path):
    """
    Processes all sheets of an Excel file, moves headers to the data as regular values,
    and combines all data into a single column DataFrame.

    Parameters:
        file_path (str): The path to the Excel file.

    Returns:
        pd.DataFrame: A single-column DataFrame containing all the values from the sheets.
    """
    # Load the Excel file
    excel_data = pd.ExcelFile(file_path)
    # Initialize an empty list to store all values
    all_values = []

    # Loop through all sheets
    for sheet_name in excel_data.sheet_names[1:]:
        # Read the sheet into a DataFrame
        df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

        # Flatten the DataFrame and add its values to the list
        all_values.extend(df.values.flatten())

    # Create a single-column DataFrame from the list
    result_df = pd.DataFrame(all_values, columns=['Values'])

    return result_df

def is_person(text,nlp_ner):
    try:
        ner_results = nlp_ner(text)
        for entity in ner_results:
            if entity['entity_group'] == 'PERS' or entity['entity_group'] == 'ORG':  # Check for person or organization entities
                return True
    except Exception as e:
        print(f"Error processing text '{text}': {e}")
    return False

def get_names_arabert(file_path):
    single_column_df = process_excel_to_single_column(file_path).dropna()
    model_name = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForTokenClassification.from_pretrained(model_name)
    nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer,aggregation_strategy="simple")
    entities_results = []
    for col in single_column_df.columns:
        for index, value in single_column_df[col].items():
            if is_person(value,nlp_ner):
                entities_results.append("PERS")
                # print(f"The value '{value}' at index {index}, column '{col}' represents a person.")
            else:
                entities_results.append("Not PERS")
                # print(f"The value '{value}' at index {index}, column '{col}' does NOT represent a person.")
    single_column_df["verdict"]=entities_results
    single_column_df = list(single_column_df[single_column_df.verdict == "PERS"].Values)
    return single_column_df

def read_pdfs_from_folder(pdf_path):
    """
    Reads PDF filenames from the input folder.

    Returns a list of PDF filenames.
    """
    pdf_files = [f for f in os.listdir(pdf_path) if f.lower().endswith('.pdf')]
    return pdf_files


def extract_tables_from_image(image):
    """
    Extracts tables as list of dictionaries from an image.

    Parameters:
    - image: The image from which to extract tables.

    Returns:
    - table_dicts: The list of extracted dictionaries from the image.
    """
    tables = extract_tables_from_image_as_dict(image,language='ara')
    table_dicts = [table['table_dict'] for table in tables]
    return table_dicts

def _get_table_meta_data(table_dict):
    columns = table_dict[0]
    num_rows, num_columns = len(table_dict), len(columns)
    meta_data = {"columns":columns, "num_row":num_rows, "num_columns": num_columns}
    return meta_data

def convert_pdf_to_images(pdf_file, pdf_path, output_path):
    """
    Converts a single PDF file to images and saves them to the output path using PyMuPDF.

    Returns a list of generated image filenames.
    """
    doc = fitz.open(os.path.join(pdf_path, pdf_file))
    image_files = []
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        pix = page.get_pixmap()
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        img = _preprocess_image(img)

        image_filename = f"{os.path.splitext(pdf_file)[0]}_page_{page_number + 1}.png"
        image_path = os.path.join(output_path, image_filename)
        img.save(image_path)
        image_files.append(image_filename)
    return image_files


def _preprocess_image(image):
    """
    Preprocesses an image before OCR.

    Returns the preprocessed image.
    """
    # Convert image to openCV format
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back to PIL image
    image = Image.fromarray(gray)
    return image

def extract_text_from_images(image_files, images_path, config):
    """
    Extracts text from a list of images.

    Returns the extracted text as a string.
    """
    full_text = ""
    for image_file in image_files:
        image_path = os.path.join(images_path, image_file)
        text = pytesseract.image_to_string(Image.open(image_path), lang=config['language'], config=config['tesseract_config'])
        full_text += text + "\n"
    return full_text


def save_extracted_text(pdf_file, text, output_text_path):
    """
    Saves the extracted text to a file in the output path.
    The file is named based on the original PDF filename and a timestamp.
    """
    # Create a timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Generate output filename with original PDF name, timestamp, and "extracted"
    output_filename = f"{os.path.splitext(pdf_file)[0]}_{timestamp}_extracted.txt"

    # Save the extracted text in the test folder
    output_file = os.path.join(output_text_path, output_filename)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)


def extract_pdf_content(file_path) -> str:
    """
    Extraire le contenu structuré d'un PDF avec Docling et Tesseract (langue arabe).
    Args:
        file_path (Path): Chemin du fichier PDF à traiter.
    Returns:
        str: Contenu extrait en format markdown.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True

    ocr_options = TesseractOcrOptions(
        force_full_page_ocr=True,  # Activer l'OCR pour chaque page
        lang=['ara', 'fra']  # Langue arabe
    )
    pipeline_options.ocr_options = ocr_options

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    doc = converter.convert(file_path).document
    return doc.export_to_markdown()

def create_vector_store(texts) -> FAISS:
    """
    Créer un magasin vectoriel avec les embeddings du texte fourni.
    Args:
        texts (List[str]): Liste de textes à indexer.
    Returns:
        FAISS: Magasin vectoriel contenant les embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2"
    )
    vector_store = FAISS.from_texts(texts, embeddings)
    return vector_store


def get_qa_chain(vector_store):
    """
    Créer une chaîne de question-réponse avec Ollama LLM et le magasin vectoriel.
    Args:
        vector_store (FAISS): Magasin vectoriel contenant les embeddings.
    Returns:
        RetrievalQA: Chaîne configurée pour répondre aux questions.
    """
    llm = OllamaLLM(model="llama3.1:8b")
    prompt_template = MAIN_PROMPT
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 10}),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )
    return qa_chain


def clean_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Loop through all the files and subdirectories in the folder
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)

            # If it's a directory, remove it recursively
            if os.path.isdir(file_path):
                shutil.rmtree(file_path)
            else:
                # If it's a file, remove it
                os.remove(file_path)
        print(f"Folder '{folder_path}' has been cleaned.")
    else:
        print(f"Folder '{folder_path}' does not exist.")
