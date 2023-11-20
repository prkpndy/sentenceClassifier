import pickle
import pandas as pd
from clean_text import clean_text
from urllib.parse import urlparse

# Change the model name to test different models
pkfile = open("model_NB.pk", "rb")
model = pickle.load(pkfile)
pkfile.close()

# Function to check if a post is asking for help or not
def is_help(post):
    post = post.strip()
    u = urlparse(post).scheme
    if(u == ''):
        post = clean_text(post)
        if(len(post.split(" ")) <= 3):
            return -1
        else:
            post = [post]
            prediction = model.predict(post)
            prob = model.predict_proba(post)
            return prediction  # prob[0][1]
    else:
        return -1

test = pd.read_csv("data/jaano_new_test2.csv")
prediction = []
actual = [1]*len(test['post'])

for string in test['post']:
    ans = is_help(string)
    print(string)
    print(ans)
    prediction.append(ans)

f = []
for i in range(len(prediction)):
    f.append(not(prediction[i]^actual[i]))

print(sum(f)/len(f))

#test = pd.read_excel("data/test.xlsx")
#print(test)
#print(type(test))
###test = pd.read_csv("data/test.csv")
# he = []
###for string in test['post_description']:
    #print(is_help(string))
    #print(string)
    #print("\n")
    #new_srt_list = re.split(r' *[\.\?!][\'"\)\]]* *', string)
    #strg = string.split("\n")
    ###
    # strg = re.split('\.|\?|\n|\!', string)
    # if((len(strg) > 5) or (len(strg) == 0)):
    #     continue
    # else:
    #     h = []
    #     for s in strg:
    #         #print(j)
    #         #print(len(j))
    #         #new_srt_list = nltk.sent_tokenize(j)
    #         #h = False
    #         #for s in new_srt_list:
    #             #h = h or is_help(s)
    #         hlp = is_help(s)
    #         #hlp = hlp[0]
    #         if(hlp>0):
    #             h.append(is_help(s))
    #             # print(s)
    #             # print(h)
    #             # print("\n")
    #     if(len(h) == 0):
    #         continue
    #     score = sum(h)/len(h)
    #     if(score>0.4):
    #         print(string, "\n", score)
#     he.append(h)
# ans = pd.DataFrame(test['post_description'], he, columns = ['post', 'label'])
# ans.to_csv("data/ans.csv")

# sent_text = nltk.sent_tokenize(text) # this gives us a list of sentences
# # now loop over each sentence and tokenize it separately
# for sentence in sent_text:
#     tokenized_text = nltk.word_tokenize(sentence)
#     tagged = nltk.pos_tag(tokenized_text)
#     print(tagged)