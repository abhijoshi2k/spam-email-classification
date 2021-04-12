# Imports

import pandas as pd
import numpy as np

import json

from bs4 import BeautifulSoup

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.sparse import data

import nltkmodules

from xgboost import XGBClassifier

from sklearn.feature_extraction.text import CountVectorizer


# Constants

PROB_SPAM = 0.310989
vocab = pd.read_csv('SpamData/01_Processing/word-by-id.csv')
prob_token_spam = np.loadtxt(
    'SpamData/03_Testing/prob-spam.txt', delimiter=' ')
prob_token_ham = np.loadtxt(
    'SpamData/03_Testing/prob-nonspam.txt', delimiter=' ')
prob_all_tokens = np.loadtxt(
    'SpamData/03_Testing/prob-all-tokens.txt', delimiter=' ')
xgb_vocab = {}
with open('./xgb-vocab.json', 'r') as fp:
    xgb_vocab = json.load(fp)
vectorizer = CountVectorizer(stop_words='english', vocabulary=xgb_vocab)


# Functions

def get_email(data, mode):
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

def is_spam(data, mode=2, classifier='manual'):

    if(classifier == 'manual'):
        message_body = data

        if(mode != 2):
            message_body = get_email(data, mode)

        clean_message = clean_message_no_html(
            message_body, stop_words=set(stopwords.words('english')))

        word_columns_df = pd.DataFrame.from_records([clean_message])
        word_columns_df.index.name = 'DOC_ID'

        word_index = pd.Index(vocab.VOCAB_WORD)

        sparse_matrix = make_sparse_matrix(word_columns_df, word_index).groupby([
            'DOC_ID', 'WORD_ID']).sum().reset_index().to_numpy()

        full_matrix = make_full_matrix(
            sparse_matrix, vocab.shape[0]).to_numpy()

        spam_email_prob = PROB_SPAM
        ham_email_prob = 1-PROB_SPAM
        # denominator = 1

        for j in range(full_matrix.shape[1]):

            if full_matrix[0, j] > 0:

                if prob_token_spam[j] > 0:
                    spam_email_prob = spam_email_prob * \
                        (prob_token_spam[j]**full_matrix[0, j])
                    if spam_email_prob == 0:
                        spam_email_prob = prev_spam
                        ham_email_prob = prev_ham
                        break

                if prob_token_ham[j] > 0:
                    ham_email_prob = ham_email_prob * \
                        (prob_token_ham[j]**full_matrix[0, j])
                    if ham_email_prob == 0:
                        spam_email_prob = prev_spam
                        ham_email_prob = prev_ham
                        break

                prev_spam = spam_email_prob
                prev_ham = ham_email_prob

                # denominator = denominator * prob_all_tokens[j]

        # print(spam_email_prob/denominator > ham_email_prob/denominator)
        print(spam_email_prob > ham_email_prob)

        # joint_log_spam = full_matrix.dot(
        #     np.log(prob_token_spam+0.000000000000001) - np.log(prob_all_tokens+0.000000000000001)) + np.log(PROB_SPAM)
        # print(joint_log_spam)

        # joint_log_ham = full_matrix.dot(
        #     np.log(prob_token_ham+0.000000000000001) - np.log(prob_all_tokens+0.000000000000001)) + np.log(1 - PROB_SPAM)
    # print(joint_log_ham)

    elif(classifier == 'xgb'):
        xgb_classifier = XGBClassifier()
        xgb_classifier.load_model('./XGB.model')

        data_list = []
        data_list.append(data)

        doc_term_matrix = vectorizer.transform(data_list)
        print(xgb_classifier.predict(doc_term_matrix)[0] == 1)


is_spam('Greeting from CSI-SIES GST,\nI hope this email finds you well.\n\nIf you have not yet submitted your abstracts, please submit it as soon as possible so that the evaluation process of the abstracts can be started and we can give you an update whether your abstract is selected or not.\n\nIn case of any other queries, you can drop in a mail on our official email ID or you can contact us:\n\n1. Sangeeth Arun, Secretary, CSI SIESGST: 9167221000\n2. Sharan Murli, Joint Secretary, CSI SIESGST: 9167754246\n\nPlease ignore this message if you have already submitted the abstract.\n\nThank You.')

is_spam('I ended up figuring out the problem. This isn\'t a problem with axios but in your react development environment the component is rendered twice if you\'re in StrictMode. The purpose of this to uncover unpredicted side-effects, it doesn\'t happen in production', classifier='xgb')

is_spam('Good day, I have a proposition involving an investment initiative to discuss with you. It will be of mutual benefit to both of us, and I believe we can handle it together, once we have a common understanding and mutual co-operation in the execution of the modalities. Should you have the capacity and interest to handle this project, please respond with your full name, address, mobile number to me. Yours sincerely, Mr. Jean Marie Kojo', classifier='xgb')

is_spam('Hello Abhishek Joshi,\nCongratulations! You have been selected for the Online Test Prep & Career Guidance Webinar hosted by Collegepond.\nAbout Collegepond:\nCollegepond is a leading test preparation, career, and admissions counselling outfit in India.  We have helped over 10,000 students secure admissions to leading universities across the globe over the last 16 years.  For more information, please refer to our website -- www.collegepond.com\nKey Takeaways from the Webinar:\n•  Upcoming fields & salary trends\n•  How to prepare for entrance exams? (GRE, GMAT, TOEFL, IELTS)\n•  How to build a strong profile/resume?\n•  Overview of the application process\n•  Planning your educational finances\nDate & Time: 1st April, 2:00 PM Onwards\nWebinar room: https://event.webinarjam.com/t/click/0nro5a4mcyngt03xtzo84ixxrpb1yma4\nSee you at the webinar!\nFor any questions or concerns feel free to contact us at info@collegepond.com', classifier='xgb')

is_spam('Dear Candidate,\n\n\nGreetings from RBCDSAI, IITM!\n\n\nIt gives us immense pleasure in inviting you to the “ 3rd RBCDSAI INDUSTRY CONCLAVE IN COLLABORATION WITH DART LAB ON FINANCIAL ANALYTICS” on 31st  March,2021.\n\n\nFor more details about the event and to register please visit\n\n\nhttps://rbcdsai.iitm.ac.in/events/3rd-conclave/\n\n\nTime:     10 am to 5,30 pm  \nDate:     31.03.2021, Wednesday\n\nVenue:  Online (Webinar details will be sent to registered participants from 5 pm to 7 pm on 30th March )\n\nThe registration is open till 30th March 2021 (6 PM IST)   \n\n\n\nSummary of the Event\n\n\n\nThe 3rd RBCDSAI Industry Conclave in collaboration with DART Lab, focuses on the special theme of Financial Analytics. The conclave is scheduled to have industry experts highlight the various applications of AI and Data Science in Finance. This covers a broad range of areas that look at financial risk as it applies to capital markets, investment management, BFSI, and household finance. Join us for this exciting one-day event, from 10am to 5:30 pm, on 31st March 2021, to listen and interact with distinguished industry leaders as they discuss important elements of Financial Analytics.\n\n\n\nLooking forward for your participation.\n\n\n\nBest regards,\n\n\n\nRBCDSAI Team', classifier='xgb')

is_spam('Ski jumping is a winter sport in which competitors aim to achieve the longest jump after descending from a specially designed ramp on their skis. Along with jump length, competitor\'s style and other factors affect the final score. Ski jumping was first contested in Norway in the late 19th century, and later spread through Europe and North America in the early 20th century. Along with cross-country skiing, it constitutes the traditional group of Nordic skiing disciplines.', classifier='xgb')
