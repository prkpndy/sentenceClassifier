# pandas
import pandas as pd
import numpy as np
# scikit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
# nltk
import nltk
from nltk.stem.snowball import SnowballStemmer
# miscellaneous
import pickle
import re
import sys
import warnings

file_path_train = "data/train_google.csv"

raw_data_train = pd.read_csv(file_path_train)

data_train = raw_data_train

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned

def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def split(string, start = 0, end = 50):
    return ' '.join(string.split()[start:end])

# cleaning training dataset
data_train['post'] = data_train['post'].str.lower()
#data_train['post'] = data_train['post'].apply(cleanHtml)
data_train['post'] = data_train['post'].apply(cleanPunc)
data_train['post'] = data_train['post'].apply(keepAlpha)

count = 0
for str in data_train['post']:
    data_train.iloc[count, 0] = split(str)
    count = count + 1

# removing stopwords
pkfile = open("stop_words.pk", "rb")
stop_words = pickle.load(pkfile)
pkfile.close()
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data_train["post"] = data_train["post"].apply(removeStopWords)

# stemming
def stemming(sentence):
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence

data_train['post'] = data_train['post'].apply(stemming)

# training and testing
txt_clf_SGD = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='modified_huber', penalty='l2',    # loss='hinge'
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None))
])

txt_clf_NB = Pipeline([
    ("vect", CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

# txt_clf_XGB = Pipeline([
#     ('vect', CountVectorizer(),
#     'tfidf', TfidfTransformer()),
#     ('clf', GradientBoostingClassifier(loss = 'exponential',
#                                        n_estimators=100,
#                                        subsample=0.6))
# ])

train_text = data_train['post']
y_train = data_train['label']

txt_clf_SGD.fit(train_text, y_train)
txt_clf_NB.fit(train_text, y_train)
#txt_clf_XGB.fit(train_text, y_train)

pkfile = open("model_SGD.pk", "wb")
pickle.dump(txt_clf_SGD, pkfile)
pkfile.close()

pkfile = open("model_NB.pk", "wb")
pickle.dump(txt_clf_NB, pkfile)
pkfile.close()

# pkfile = open("model_XGB.pk", "wb")
# pickle.dump(txt_clf_XGB, pkfile)
# pkfile.close()

#OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)

####
count_vect = CountVectorizer()
tfidf_trans = TfidfTransformer()

x_count = count_vect.fit_transform(train_text)
x_train = tfidf_trans.fit_transform(x_count)

clf = GradientBoostingClassifier(loss = 'exponential',
                                 n_estimators=10,
                                 subsample=0.6)

clf.fit(x_train, y_train)

pkfile = open("XGB_CV.pk", "wb")
pickle.dump(count_vect, pkfile)
pkfile.close()

pkfile = open("XGB_TFIDF_trans.pk", "wb")
pickle.dump(tfidf_trans, pkfile)
pkfile.close()

pkfile = open("XGB_clf.pk", "wb")
pickle.dump(clf, pkfile)
pkfile.close()

