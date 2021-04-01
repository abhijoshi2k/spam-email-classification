# Imports

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import nltkmodules


# Constants

PROB_SPAM = 0.310989
vocab = pd.read_csv('SpamData/01_Processing/word-by-id.csv')
prob_token_spam = np.loadtxt(
    'SpamData/03_Testing/prob-spam.txt', delimiter=' ')
prob_token_ham = np.loadtxt(
    'SpamData/03_Testing/prob-nonspam.txt', delimiter=' ')
prob_all_tokens = np.loadtxt(
    'SpamData/03_Testing/prob-all-tokens.txt', delimiter=' ')


# Functions

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


# Main Function

def is_spam(data, mode):
    message_body = get_email(data, mode)

    clean_message = clean_message_no_html(
        message_body, stop_words=set(stopwords.words('english')))

    word_columns_df = pd.DataFrame.from_records([clean_message])
    word_columns_df.index.name = 'DOC_ID'

    word_index = pd.Index(vocab.VOCAB_WORD)

    sparse_matrix = make_sparse_matrix(word_columns_df, word_index).groupby([
        'DOC_ID', 'WORD_ID']).sum().reset_index().to_numpy()

    full_matrix = make_full_matrix(sparse_matrix, vocab.shape[0]).to_numpy()

    joint_log_spam = full_matrix.dot(
        np.log(prob_token_spam) - np.log(prob_all_tokens)) + np.log(PROB_SPAM)
    print(joint_log_spam)

    joint_log_ham = full_matrix.dot(
        np.log(prob_token_ham) - np.log(prob_all_tokens)) + np.log(1 - PROB_SPAM)
    print(joint_log_ham)


is_spam('Dear Mr./Ms./Mrs. {Recipient\'s Name},\nThis is with reference to your job requirement on {portal name} for the role of Sales Manager. I truly believe that my qualifications and experience make me a perfect candidate for the job.\nI completed my MBA in Sales and Marketing from {Institute Name}. I have worked as an Area Sales Manager and Assistant Marketing Manager at {Company Name}. During my stint as Area Sales Manager, I conceptualised and executed a Customer Engagement Program that resulted in higher sales. As Assistant Marketing Manager, I worked on the planning and execution of a new product launch. With 4 years of experience in B2B sales and marketing, I have an in-depth understanding of the process. I am confident that I will be the right fit for the job.\n\nI have attached my CV to the email for your reference. Please have a look at it.\n\nI hope to meet you and discuss this opportunity further. Thank you for considering my application for the role.', 2)
