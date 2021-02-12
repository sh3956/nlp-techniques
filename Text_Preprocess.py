import pandas as pd
import numpy as np

FileName = 'IMDB Dataset.csv'
TextName = 'review.txt'

def preprocess_data():
    """ preprocess the dataset to txt
    for later use
    returns: a string of review text
    """
    read_data()
    with open(TextName, 'rt') as file:
        text = file.read()
        text.replace('<br />', '')
        words = text.split()
        final_string = ''
        for w in words:
            w = remove_punctuation(w)
            w = normalize_case(w)
            final_string += w
            final_string += " "

    return final_string

def read_data():
    """ reads csv file and saves the text
    """
    review_df = pd.read_csv(FileName)[:50000]
    review = review_df['review']
    np.savetxt(TextName, review, fmt='%s', delimiter="\t")

def remove_punctuation(word):
    """ remove the punctuation of the txt file
    params:
        word(string): current word
    returns:
        if the word is a punctuation, return None
    """
    import string
    table = str.maketrans('', '', string.punctuation)
    stripped = word.translate(table)
    return stripped

def normalize_case(word):
    """ normalize the words to small case
    """
    return word.lower()


preprocess_data()