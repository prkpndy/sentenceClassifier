import re
import sys
import warnings
from nltk.stem.snowball import SnowballStemmer
import pickle
import googletrans as gt

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def split(string, start = 0, end = 50):
    return ' '.join(string.split()[start:end])

def deEmojify(string):
    emoji_pattern = re.compile("["
                            u"\U0001F600-\U0001F64F"  # emoticons
                            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                            u"\U0001F680-\U0001F6FF"  # transport & map symbols
                            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            u"\U00002702-\U000027B0"
                            u"\U000024C2-\U0001F251"
                            "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)


def clean_text(sentence):
    # Translating
    translator = gt.Translator()
    sentence_translated = translator.translate(deEmojify(sentence)).text

    # Removing html tags
    cleanr = re.compile('<.*?>')
    cleaned = re.sub(cleanr, ' ', str(sentence_translated))

    # Removing punctuations
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',cleaned)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")

    # Removing non-alphabetic characters
    alpha_sent = ""
    for word in cleaned.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    cleaned = alpha_sent.strip()

    # Removing stopwords
    pkfile = open("stop_words.pk", "rb")
    stop_words = pickle.load(pkfile)
    pkfile.close()
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    cleaned = re_stop_words.sub(" ", cleaned)

    # Stemming
    stemmer = SnowballStemmer("english")
    stemSentence = ""
    for word in cleaned.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    cleaned = stemSentence.strip()

    # Splitting
    cleaned = split(cleaned)

    return cleaned