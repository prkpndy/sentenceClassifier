create_model.py --> creates the model and stores it in model.pk file
create_stopwords.py --> creates a set of stopwords and stores it in stop_words.pk file
clean_text.py --> preprocess the text
inference.py --> takes a sentence and gives the output as help or not_help (change the name of the model to model_SGD.pk or model_NB.pk)
inference_xgboost.py --> takes a sentence and gives the output as help or not_help using xgboost model


model_SGD.pk --> stores the SGD model
model_NB.pk --> stores NB model
XGB_clf.pk --> stores XGB classifier
stop_words.pk --> stores the stopwords

data folder contains the data for training, evaluation and testing data
train_google.csv --> training data 1
train_itslack.csv --> training data 2
test.xlsx --> data made manually



