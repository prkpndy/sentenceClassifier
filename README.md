create_model.py --> creates the model and stores it in model.pk file
create_stopwords.py --> creates a set of stopwords and stores it in stop_words.pk file
clean_text.py --> preprocess the text
inference.py --> takes a sentence and gives the output

model.pk --> stores the model
stop_words.pk --> stores the stopwords

data folder contains the data for training, evaluation and testing data
training.csv --> contains the training data (mixed stackoverflow and toxic_language dataset)
eval.csv --> contains the test data (mixed stackoverflow and toxic_language dataset)
test.csv --> contains the posts from jaano app
test.xlsx --> data made manually


