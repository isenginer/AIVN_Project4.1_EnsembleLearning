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


def category_numerical(category: str=""):
    """
    function to return the value of categorical string in list
    :param category: the category string
    :return: value of categorical string in lis/t
    """
    category_list = ["astro-ph", "cond-mat", "cs", "math", "physics"]
    category_dict = dict(zip(category_list, range(1, 6)))
    if category not in category_list:
        return 0
    else:
        return category_dict[category]


if __name__ == "__main__":
    text = "cond-mat"
    print(category_numerical(text))