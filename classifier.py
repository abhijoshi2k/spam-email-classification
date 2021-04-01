import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltkmodules

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
    """

    nr_cols = df.shape[1]
    word_set = set(indexed_words)
    dict_list = []

    for j in range(nr_cols):

        word = df.iat[0, j]
        if word in word_set:
            doc_id = df.index[0]
            word_id = indexed_words.get_loc(word)

            item = {'DOC_ID': doc_id, 'OCCURENCE': 1, 'WORD_ID': word_id}

            dict_list.append(item)

    return pd.DataFrame(dict_list)


def make_full_matrix(sparse_matrix, nr_words, doc_idx=0, word_idx=1, freq_idx=2):
    """
    Form a full matrix full matrix from a sparse matrix. Return a pandas DataFrame.
    Keywords arguments:
    sparse_matrix -- numpy array
    nr_words -- size of the vocabulary. Total number of tokens.
    doc_idx -- position of the document id in sparse matrix. Default: 1st column
    word_idx -- position of the word id in sparse matrix. Default: 2nd column
    freq_idx -- position of occurrence of word in sparse matrix. Default 3rd column
    """

    column_names = ['DOC_ID'] + list(range(0, nr_words))
    doc_id_names = np.unique(sparse_matrix[:, doc_idx])

    full_matrix = pd.DataFrame(index=doc_id_names, columns=column_names)
    full_matrix.fillna(value=0, inplace=True)

    for i in range(sparse_matrix.shape[0]):
        doc_nr = sparse_matrix[i][doc_idx]
        word_id = sparse_matrix[i][word_idx]
        occurrence = sparse_matrix[i][freq_idx]

        full_matrix.at[doc_nr, 'DOC_ID'] = doc_nr
        full_matrix.at[doc_nr, word_id] = occurrence

    full_matrix.set_index('DOC_ID', inplace=True)

    return full_matrix


def is_spam(data, mode):
    message_body = get_email(data, mode)

    clean_message = clean_message_no_html(
        message_body, stop_words=set(stopwords.words('english')))

    word_columns_df = pd.DataFrame.from_records([clean_message])
    word_columns_df.index.name = 'DOC_ID'

    word_index = pd.Index(vocab.VOCAB_WORD)

    sparse_matrix = make_sparse_matrix(word_columns_df, word_index).groupby([
        'DOC_ID', 'WORD_ID']).sum().reset_index()

    sparse_matrix = sparse_matrix.to_numpy()

    full_matrix = make_full_matrix(sparse_matrix, vocab.shape[0])
    print(full_matrix)


is_spam('Hello hello boy http http', 2)