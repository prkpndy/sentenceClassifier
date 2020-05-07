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

pkfile = open("model.pk", "rb")
model = pickle.load(pkfile)
pkfile.close()

def is_help(post):
    post = clean_text(post)

    post = [post]
    prediction = model.predict(post)
    prob = model.predict_proba(post)
    return prob


test = pd.read_excel("test.xlsx")
#print(test)
#print(type(test))

for string in test['posts']:
    print(is_help(string))
    #print(string)
