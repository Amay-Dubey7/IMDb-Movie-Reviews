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


#Data cleaning
from nltk.corpus import stopwords
stop = stopwords.words('english')
dataset['review'] = dataset['review'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

dataset['review'] = dataset['review'].apply(lambda x: " ".join(x.lower() for x in x.split()))
dataset['review'] = dataset['review'].str.replace('[^\w\s]','') #punctuation

#Tokenizing
from nltk.tokenize import RegexpTokenizer
#instantiate tokenizer
tokenizer = RegexpTokenizer(r'\w+')
dataset['review'] = dataset['review'].apply(lambda x: tokenizer.tokenize(x))

#Stemming
porter = PorterStemmer()
def word_stemmer(text):
    stem_text = " ".join([porter.stem(i) for i in text])
    return stem_text
dataset['review'] = dataset['review'].apply(lambda x: word_stemmer(x))

#Bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 6000)
X = cv.fit_transform(dataset['review']).toarray()
y = dataset.iloc[:, 1].values


#splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)


# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results for Naive bayes
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
cm_nb = confusion_matrix(y_test, y_pred)
print("Gaussian Naive Bayes model accuracy(in %):",accuracy_score(y_test, y_pred)*100)
print("Gaussian Naive Bayes f1_score (in %):",f1_score(y_test, y_pred)*100)


#Fitting logistic_regression to the training set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=42)
lr.fit(X_train,y_train)

#Predicting the test set results for Logistic Regression
y_pred_lr = lr.predict(X_test)
print("Logistic Regression model accuracy(in %):",accuracy_score(y_test, y_pred_lr)*100)
print("Logistic Regression f1_score(in %):",f1_score(y_test, y_pred_lr)*100)


#Fitting Random Forest classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 100)
rf.fit(X_train,y_train)

#Predicting the test set results for Random Forest Classifier
y_pred_fr = rf.predict(X_test)
print("Random Forest model accuracy(in %):",accuracy_score(y_test, y_pred_fr)*100)
print("Random Forest f1_score(in %):",f1_score(y_test, y_pred_fr)*100)


#Fitting SVM to training set
from sklearn.svm import SVC
svc= SVC(kernel = 'linear', random_state = 0)
svc.fit(X_train, y_train)

#Predicting the test set results for SVM
y_pred_svc = svc.predict(X_test)
print("SVM model accuracy(in %):",accuracy_score(y_test, y_pred_svc)*100)
print("SVM f1_score(in %):",f1_score(y_test, y_pred_svc)*100)


#LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

maxlen = 130
X_train = pad_sequences(X_train,maxlen = maxlen)
X_test = pad_sequences(X_test,maxlen = maxlen)

#Architecture
max_features = 6000
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
y_pred_lstm = (prediction>0.5)
print("LSTM model accuracy(in %):",accuracy_score(y_test, y_pred_lstm)*100)
print("LSTM f1_score(in %):",f1_score(y_test, y_pred_lstm)*100)
