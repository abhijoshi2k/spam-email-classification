import pandas as pd
import numpy as np

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup

TOKEN_SPAM_PROB_FILE = 'SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/03_Testing/prob-all-tokens.txt'


def get_email(data, mode):
    if mode == 2:
        return data

    else:
        body = False
        lines = []

        for line in data:
            if body or mode == 1:
                lines.append(line)
            elif line == '\n':
                body = True

        return '\n'.join(lines)


def clean_message_no_html(message, stop_words):

    # Remove HTML Tags
    soup = BeautifulSoup(message, 'html.parser')
    cleaned_text = soup.get_text()

    # Converts to lower case and splits up the words
    words = word_tokenize(cleaned_text.lower())

    filtered_words = []

    for word in words:
        # Removes stop words and punctuation
        if word not in stop_words and word.isalpha():
            filtered_words.append(PorterStemmer().stem(word))

    return filtered_words


def is_spam(data, mode):
    message_body = get_email(data, mode)

    email = clean_message_no_html(
        message_body, stop_words=set(stopwords.words('english')))
