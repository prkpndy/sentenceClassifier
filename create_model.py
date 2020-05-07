import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier
import nltk
from nltk.stem.snowball import SnowballStemmer
import pickle
import re
import sys
import warnings

file_path_train = "data/train.csv"
file_path_eval = "data/eval.csv"
file_path_test = "data/test.csv"

raw_data_train = pd.read_csv(file_path_train)
raw_data_eval = pd.read_csv(file_path_eval)
raw_data_test = pd.read_csv(file_path_test)

data_train = raw_data_train
data_eval = raw_data_eval
data_test = raw_data_test

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

# cleaning evalution dataset
data_eval['post'] = data_eval['post'].str.lower()
#data_eval['post'] = data_eval['post'].apply(cleanHtml)
data_eval['post'] = data_eval['post'].apply(cleanPunc)
data_eval['post'] = data_eval['post'].apply(keepAlpha)

count = 0
for str in data_eval['post']:
    data_eval.iloc[count, 0] = split(str)
    count = count + 1

# cleaning testing dataset
data_test['post_description'] = data_test['post_description'].str.lower()
#data_test['post_description'] = data_test['post_description'].apply(cleanHtml)
data_test['post_description'] = data_test['post_description'].apply(cleanPunc)
data_test['post_description'] = data_test['post_description'].apply(keepAlpha)

count = 0
for str in data_test['post_description']:
    data_test.iloc[count, 0] = split(str)
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
data_eval["post"] = data_eval["post"].apply(removeStopWords)
data_test["post_description"] = data_test["post_description"].apply(removeStopWords)

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
data_eval['post'] = data_eval['post'].apply(stemming)
data_test['post_description'] = data_test['post_description'].apply(stemming)

# training and testing
txt_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LogisticRegression(solver='sag'), n_jobs=-1))
])

train_text = data_train['post']
y_train = data_train['label']
txt_clf.fit(train_text, y_train)

pkfile = open("model.pk", "wb")
pickle.dump(txt_clf, pkfile)
pkfile.close()

eval_text = data_eval['post']
test_text = data_test['post_description']
y_eval = data_eval['label']
prediction_eval = txt_clf.predict(eval_text)
prediction_test = txt_clf.predict(test_text)
print('Evaluation accuracy is {}'.format(accuracy_score(y_eval, prediction_eval)))

#SGDClassifier(loss='modified_huber', penalty='l2',    # loss='hinge'
#                    alpha=1e-3, random_state=42,
#                    max_iter=5, tol=None)