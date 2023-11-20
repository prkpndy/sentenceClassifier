# pandas
import pandas as pd

# scikit learn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics

# nltk
from nltk.stem.snowball import SnowballStemmer

# miscellaneous
import pickle
import re
import sys
import warnings

# ------------ Reading the data on which to train the model
file_path_train = "data/train_google.csv"
raw_data_train = pd.read_csv(file_path_train)  # will take the first row as header
data_train = raw_data_train

if not sys.warnoptions:
    warnings.simplefilter("ignore")

# Function to remove HTML from the sentence
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext

# Function the clean the sentence of any punctuation or special character
def cleanPunc(sentence):
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

# Reduce the string from start to end
def split(string, start = 0, end = 50):
    return ' '.join(string.split()[start:end])

# ------------ Cleaning training dataset
data_train['post'] = data_train['post'].str.lower()
# data_train['post'] = data_train['post'].apply(cleanHtml)
data_train['post'] = data_train['post'].apply(cleanPunc)
data_train['post'] = data_train['post'].apply(keepAlpha)
count = 0
for str in data_train['post']:
    data_train.iloc[count, 0] = split(str)
    count = count + 1

# ------------ Removing stopwords
pkfile = open("stop_words.pk", "rb")
stop_words = pickle.load(pkfile)
pkfile.close()
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)

# Function to remove the stopwords from the sentence
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)

data_train["post"] = data_train["post"].apply(removeStopWords)

# ------------ Stemming - Reduce the base word to its stem word using snowball stemmer from NLTK
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

# pkfile = open("model_NB.pk", "rb")
# txt_clf_NB = pickle.load(pkfile)
# pkfile.close()
# pkfile = open("model_SGD.pk", "rb")
# txt_clf_SGD = pickle.load(pkfile)
# pkfile.close()

# ------------ Training
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

# Getting the training and testing set
train_text = data_train['post']
y_train = data_train['label']
train_text, test_text, y_train, y_test = train_test_split(train_text, y_train, test_size = 0.2, random_state = 42)

# Training
txt_clf_SGD.fit(train_text, y_train)
txt_clf_NB.fit(train_text, y_train)
# txt_clf_XGB.fit(train_text, y_train)

# Storing the models
pkfile = open("model_SGD.pk", "wb")
pickle.dump(txt_clf_SGD, pkfile)
pkfile.close()

pkfile = open("model_NB.pk", "wb")
pickle.dump(txt_clf_NB, pkfile)
pkfile.close()

# pkfile = open("model_XGB.pk", "wb")
# pickle.dump(txt_clf_XGB, pkfile)
# pkfile.close()

# OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1)

# count_vect = CountVectorizer()
# tfidf_trans = TfidfTransformer()

# x_count = count_vect.fit_transform(train_text)
# x_train = tfidf_trans.fit_transform(x_count)

# clf = GradientBoostingClassifier(loss = 'exponential',
#                                  n_estimators=100,
#                                  subsample=0.6)

# clf.fit(x_train, y_train)

# pkfile = open("XGB_CV.pk", "wb")
# pickle.dump(count_vect, pkfile)
# pkfile.close()

# pkfile = open("XGB_TFIDF_trans.pk", "wb")
# pickle.dump(tfidf_trans, pkfile)
# pkfile.close()

# pkfile = open("XGB_clf.pk", "wb")
# pickle.dump(clf, pkfile)
# pkfile.close()

# ------------ Testing

# Accuracy for SGD Classifier
print("Training Accuracy for SGD Classifier = ",accuracy_score(y_train, txt_clf_SGD.predict(train_text)))
pred_SGD = txt_clf_SGD.predict(test_text)
print("Test Accuracy for SGD Classifier = ",accuracy_score(y_test,pred_SGD))
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_SGD, pos_label=1)
print("AUC for SGD Classifier = ", metrics.auc(fpr, tpr))
print("F1 Score for SGD Classifier = ", metrics.f1_score(y_test, pred_SGD))

# Accuracy for NB Classifier
print("Training Accuracy for NB Classifier = ",accuracy_score(y_train, txt_clf_NB.predict(train_text)))
pred_NB = txt_clf_NB.predict(test_text)
print("Test Accuracy for NB Classifier = ",accuracy_score(y_test,pred_NB))
fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_NB, pos_label=1)
print("AUC for NB Classifier = ", metrics.auc(fpr, tpr))
print("F1 Score for NB Classifier = ", metrics.f1_score(y_test, pred_NB))

# Accuracy for xgboost
# x_count = count_vect.transform(test_text)
# x_test = tfidf_trans.transform(x_count)
# pred_xgb = clf.predict(x_test)
# print("Training Accuracy for xgboost Classifier = ",accuracy_score(y_train, clf.predict(x_train)))
# print("Test Accuracy for xgboost Classifier = ",accuracy_score(y_test,pred_xgb))
# fpr, tpr, thresholds = metrics.roc_curve(y_test, pred_xgb, pos_label=1)
# print("AUC for xgboost Classifier = ", metrics.auc(fpr, tpr))
# print("F1 Score for xgboost Classifier = ", metrics.f1_score(y_test, pred_xgb))
