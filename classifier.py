import pandas as pd
import numpy as np

import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from bs4 import BeautifulSoup
from pandas.core.arrays import sparse

TOKEN_SPAM_PROB_FILE = 'SpamData/03_Testing/prob-spam.txt'
TOKEN_HAM_PROB_FILE = 'SpamData/03_Testing/prob-nonspam.txt'
TOKEN_ALL_PROB_FILE = 'SpamData/03_Testing/prob-all-tokens.txt'

vocab = pd.read_csv('SpamData/01_Processing/word-by-id.csv')


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


def make_sparse_matrix(df, indexed_words):
    """
    Returns Sparse Matrix as DataFrame.

    df: A DataFrame with words in the columns with a document id as an index (X_train or X_test)
    indexed_words: Index of words ordered by word id
    labels: Category as a Series (y_train or y_test)
    """

    nr_rows = df.shape[0]
    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []

    for i in range(nr_rows):
        for j in range(nr_cols):

            word = df.iat[i, j]
            if word in word_set:
                doc_id = df.index[i]
                word_id = indexed_words.get_loc(word)

                item = {'DOC_ID': doc_id, 'OCCURENCE': 1, 'WORD_ID': word_id}

                dict_list.append(item)

    return pd.DataFrame(dict_list)


def is_spam(data, mode):
    message_body = get_email(data, mode)

    clean_message = clean_message_no_html(
        message_body, stop_words=set(stopwords.words('english')))

    word_columns_df = pd.DataFrame.from_records([clean_message])
    word_columns_df.index.name = 'DOC_ID'

    word_index = pd.Index(vocab.VOCAB_WORD)

    sparse_matrix = make_sparse_matrix(word_columns_df, word_index).groupby([
        'DOC_ID', 'WORD_ID']).sum().reset_index()

    print(sparse_matrix)
