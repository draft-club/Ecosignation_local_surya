import re
from datetime import datetime
import pandas as pd


def process_text_file(file_path):
    """
    Reads a text file, splits its content into words, and returns the words as a list.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        words = file.read().split()
    return words


def remove_punctuation(words):
    """
    Removes punctuation from a list of words.
    """
    punctuation_regex = r"[^\w\s]"  # Matches any character that is not a word or whitespace
    return [re.sub(punctuation_regex, "", word) for word in words]


def extract_matching_fields_with_context(words, regex, description, matches_with_context, field_name, is_regex, is_keyword):
    """
    Extracts words matching the given regex, stores them in a dictionary with their indexes,
    and prints context based on the field type:
    - LOI: Return only the matched word.
    - Year: Return the matched word plus the previous two words.
    - Amount ("درهم" or "DH"): Return the matched word and the three previous words.
    - Area ("متر"): Return the matched word and the three previous words.
    - Others: Return three previous, the matched word, and three following elements.
    Adds a boolean flag for the specific field type.
    """
    print(f"\n--- Matching fields for: {description} ---")

    for index, word in enumerate(words):
        if re.match(regex, word):
            # Determine context based on field type
            if field_name == "is_loi":
                context = [word]
            elif field_name == "is_year":
                context = words[max(0, index - 2):index + 1]
            elif field_name in ["is_amount", "is_area"]:
                context = words[max(0, index - 3):index + 1]
            else:
                context = words[max(0, index - 3):index + 1] + words[index + 1:index + 4]

            # Store match and its index with the field flag
            matches_with_context[index] = {
                "word": word,
                "context": context,
                field_name: 1,  # Set the flag for this field type
                "is_regex": is_regex,  # Set regex indicator
                "is_keyword": is_keyword,  # Set keyword indicator
            }

            # Print the match with context
            context_str = " ".join(context)
            print(f"Matched Element: {word}, Index: {index}, Context: {context_str}")


# Main script
if __name__ == "__main__":
    # Replace 'your_file.txt' with the path to your text file
    file_path = 'output_surya_commune.txt'

    # Step 1: Read the words from the file
    all_text_list = process_text_file(file_path)

    # Step 2: Remove punctuation
    cleaned_elements = remove_punctuation(all_text_list)

    # Step 3: Define the LOI regex
    loi_regex = r"^\d\.\d{2}\.\d{3}$"

    # Step 4: Define the year regex for years between 1980 and the current year
    current_year = datetime.now().year
    year_regex = fr"^(19(8[0-9]|9[0-9])|20(0[0-9]|1[0-9]|2[0-{current_year % 10}]))$"

    # Step 5: Define the Arabic words to match
    arabic_word_1 = "الجريدة"
    arabic_word_2 = "عدد"
    arabic_word_3 = "الرسمية"
    arabic_word_amount = "درهم"
    keyword_amount = "DH"
    arabic_word_area = "متر"
    arabic_word_decision_1 = "قرار"
    arabic_word_decision_2 = "مقرر"
    arabic_word_decision_3 = "حكم"

    # Step 6: Create a shared dictionary to store matches
    matches_with_context = {}

    # Step 7: Extract and print matching fields for LOI regex (using uncleaned list)
    extract_matching_fields_with_context(all_text_list, loi_regex, "LOI pattern (e.g., 2.22.645)", matches_with_context, "is_loi", is_regex=1, is_keyword=0)

    # Step 8: Extract and print matching fields for year regex (using cleaned list)
    extract_matching_fields_with_context(cleaned_elements, year_regex, f"Valid years between 1980 and {current_year}", matches_with_context, "is_year", is_regex=1, is_keyword=0)

    # Step 9: Look for the Arabic words "جريدة", "عدد", "الرسمية", "درهم", "DH", "متر", "قرار", "مقرر", and "حكم"
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_1}\b", f"Arabic word: {arabic_word_1}", matches_with_context, "is_جريدة", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_2}\b", f"Arabic word: {arabic_word_2}", matches_with_context, "is_عدد", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_3}\b", f"Arabic word: {arabic_word_3}", matches_with_context, "is_الرسمية", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_amount}\b", f"Arabic word: {arabic_word_amount}", matches_with_context, "is_amount", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{keyword_amount}\b", f"Keyword: {keyword_amount}", matches_with_context, "is_amount", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_area}\b", f"Arabic word: {arabic_word_area}", matches_with_context, "is_area", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_decision_1}\b", f"Arabic word: {arabic_word_decision_1}", matches_with_context, "is_قرار", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_decision_2}\b", f"Arabic word: {arabic_word_decision_2}", matches_with_context, "is_مقرر", is_regex=0, is_keyword=1)
    extract_matching_fields_with_context(all_text_list, rf"\b{arabic_word_decision_3}\b", f"Arabic word: {arabic_word_decision_3}", matches_with_context, "is_حكم", is_regex=0, is_keyword=1)

    # Step 10: Ensure all boolean fields are present in the dictionary
    for match in matches_with_context.values():
        match.setdefault("is_loi", 0)
        match.setdefault("is_year", 0)
        match.setdefault("is_جريدة", 0)
        match.setdefault("is_عدد", 0)
        match.setdefault("is_الرسمية", 0)
        match.setdefault("is_amount", 0)
        match.setdefault("is_area", 0)
        match.setdefault("is_قرار", 0)
        match.setdefault("is_مقرر", 0)
        match.setdefault("is_حكم", 0)
        match.setdefault("is_regex", 0)
        match.setdefault("is_keyword", 0)

    # Step 11: Convert the dictionary into a DataFrame
    df = pd.DataFrame.from_dict(matches_with_context, orient="index")
    df["context"] = df["context"].apply(lambda x: " ".join(x))  # Convert context list to string for readability

    # Step 12: Sort the DataFrame by index
    df = df.sort_index()

    # Step 13: Print the sorted DataFrame
    print("\n--- Matches DataFrame (Sorted by Index) ---")
    print(df.to_string())
