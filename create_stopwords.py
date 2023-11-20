import nltk
from nltk.corpus import stopwords
import pickle
# nltk.download('stopwords')
# print(type(stopwords.words("english")))

stop_words = set(stopwords.words('english'))
new_stop_words = ['zero','one','two','three','four','five','six','seven','eight','nine','ten',
                  'also','across','among','beside','within', 'r', 'function', 'code',
                  'library', 'libraries']
not_stopwords = {'which', 'when', 'how', 'but', 'should', 'where', 'who', 'whom', 'what', 'help', 'kindly', 'urgent'}

stop_words = stop_words.union(new_stop_words)
stop_words = set([word for word in stop_words if word not in not_stopwords])

pkfile = open("stop_words.pk", "wb")
pickle.dump(stop_words, pkfile)
pkfile.close()
