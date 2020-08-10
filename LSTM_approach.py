#importing libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#importing the dataset
dataset = pd.read_csv('IMDB Dataset.csv')

#now encoding the sentiment labels : 0 is negative, 1 is positive
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
dataset['sentiment'] = label_encoder.fit_transform(dataset['sentiment'])

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english")) 
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r'[^\w\s]','',text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if not word in stop_words]
    text = " ".join(text)
    return text

dataset['review'] = dataset.review.apply(lambda x: clean_text(x))

dataset.review.apply(lambda x: len(x.split(" "))).mean()
X = dataset['review']
y = dataset.iloc[:, 1].values

#LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

max_features = 6000
tokenizer = Tokenizer(num_words = max_features)
tokenizer.fit_on_texts(X)
list_tokenized_train = tokenizer.texts_to_sequences(X)

maxlen = 130
X = pad_sequences(list_tokenized_train,maxlen = maxlen)

#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#Architecture
model = Sequential()
model.add(Embedding(max_features,128))
model.add(Bidirectional(LSTM(32,return_sequences = True)))
model.add(GlobalMaxPool1D())
model.add(Dense(20,activation = 'relu'))
model.add(Dropout(0.05))
model.add(Dense(1,activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.fit(X_train,y_train,batch_size = 100,epochs = 3, validation_split = 0.2)

#predicting the test results
prediction = model.predict(X_test)
y_pred = (prediction>0.5)
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score
accuracy_score = accuracy_score(y_test,y_pred)
f1_score = f1_score(y_test,y_pred)
confusion_matrix(y_pred, y_test)