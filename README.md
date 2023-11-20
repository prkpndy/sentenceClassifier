# Files

`create_model.py` --> creates the model and stores it in model.pk file
`create_stopwords.py` --> creates a set of stopwords and stores it in stop_words.pk file
`clean_text.py` --> preprocess the text
`inference.py` --> takes a sentence and gives the output as help or not_help (change the name of the model to model_SGD.pk or model_NB.pk)
`inference_xgboost.py` --> takes a sentence and gives the output as help or not_help using xgboost model

`model_SGD.pk` --> stores the SGD model
`model_NB.pk` --> stores NB model
`XGB_clf.pk` --> stores XGB classifier
`stop_words.pk` --> stores the stopwords

data folder contains the data for training, evaluation and testing data
`train_google.csv` --> training data 1
`train_itslack.csv` --> training data 2
`test.xlsx` --> data made manually

# How it works

* First the models are created using `./create_model.py`. It will store the models in the root folder.
* To get the predictions, use `./inference.py`.
* For both training and predictions preprocessing of sentences is done as follows:
  * convert everything to lower case
  * remove any HTML
  * remove all emojis, numbers and punctuations
  * remove all stopwords
  * perform stemming

# Results

Training Accuracy for SGD Classifier =  1.0
Test Accuracy for SGD Classifier =  0.9640718562874252
AUC for SGD Classifier =  0.96875
F1 Score for SGD Classifier =  0.9594594594594594

Training Accuracy for NB Classifier =  0.9954954954954955
Test Accuracy for NB Classifier =  0.9640718562874252
AUC for NB Classifier =  0.9650821596244132
F1 Score for NB Classifier =  0.9583333333333334

