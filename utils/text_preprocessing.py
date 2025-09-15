"""
This module is support for text preprocessing. It will be applied as utilities for main notebooks of other project
"""

import re

def character_preprocessing(text):
    """
    Function to preprocess abstracts: remove all special characters
    :param abstract_text: the text to be preprocessed
    :return: text after preprocessing
    """
    # remove the enter space
    text = text.strip().replace('\n', ' ')
    # remove the special letters
    text = re.sub(r'[^\w\s]', '', text)
    # remove digit
    text = re.sub(r'\d+', '', text)
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # lower case
    text = text.lower()
    return text

def category_processing(text):
    """
    Function to preprocess categories: collect only first part
    :param text: catergories to be processed
    :return: text after preprocessing
    """
    text_extracted = text.split(" ")[0]
    text_splitted = text_extracted.replace(".", " ")
    text_splitted = text_splitted.replace(",", " ")
    text_category = text_splitted.split(" ")[0]
    return text_category

if __name__ == "__main__":
    """
    Test function of text preprocessing and character preprocessing
    """
    text = "phsic.gen sabidcb"
    print(category_processing(text))