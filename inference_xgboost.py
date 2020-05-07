import pickle
import pandas as pd
import sklearn as skl
import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import googletrans
from googletrans import Translator
import re
import sys
import warnings
from clean_text import *

pkfile = open("XGB_clf.pk", "rb")
clf = pickle.load(pkfile)
pkfile.close()

pkfile = open("XGB_CV.pk", "rb")
cv = pickle.load(pkfile)
pkfile.close()

pkfile = open("XGB_TFIDF_trans.pk", "rb")
tfidf_trans = pickle.load(pkfile)
pkfile.close()

def is_help(post):
    post = clean_text(post)
    post = [post]
    x_count = cv.transform(post)
    x_test = tfidf_trans.transform(x_count)
    prediction = clf.predict(x_test)
    prob = clf.predict_proba(x_test)
    return prediction


test = pd.read_excel("data/test.xlsx")
#print(test)
#print(type(test))

for string in test['posts']:
    # new_srt_list = re.split(r' *[\.\?!][\'"\)\]]* *', string)
    # h = False
    # for s in new_srt_list:
    #     h = h or is_help(s)
    print(is_help(string))
