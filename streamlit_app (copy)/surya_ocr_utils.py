import json
import os
from PIL import Image
from surya.model.detection.model import load_model as load_det_model, load_processor as load_det_processor
from surya.model.recognition.model import load_model as load_rec_model
from surya.model.recognition.processor import load_processor as load_rec_processor
from surya.ocr import run_ocr


def generate_txt_from_json(json_data):
    txt_content = ''

    for page, items in json_data.items():
        # Process each page's text lines
        for item in items:
            data = json.loads(item)
            text_lines = data['text_lines']

            for line in text_lines:
                text = line['text']

                # Append the text to the output content
                txt_content += f'{text}\n'

    return txt_content


def save_txt_to_file(txt_content, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(txt_content)


# Usage

def from_surya_to_txt(image_file, output_txt_file, interim_json_file):
    from_surya_to_json(image_file, interim_json_file)
    # Read JSON data from file
    with open(interim_json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # Generate the text content
    txt_content = generate_txt_from_json(json_data)

    # Save the generated text to a file
    save_txt_to_file(txt_content, output_txt_file)
    print(f"Text data has been dumped into {output_txt_file}")

    return txt_content


def from_surya_to_json(image_file, output_file):
    langs = ["ar"]
    det_processor, det_model = load_det_processor(), load_det_model()
    rec_model, rec_processor = load_rec_model(), load_rec_processor()

    # Display file name
    print(image_file)
    image = Image.open(image_file)
    # Create a JSON object from OCR results on the single image
    ocr_result_json = run_ocr([image], [langs], det_model, det_processor, rec_model, rec_processor)[0].model_dump_json()

    # Construct a JSON file with document integer id as the key and the JSON object as the value
    combined_json = {"0": [ocr_result_json]}

    # Open the file in write mode and dump the JSON into it
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(combined_json, json_file, ensure_ascii=False, indent=4)
    print(f"JSON data has been dumped into {output_file}")
