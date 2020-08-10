#importing libraries
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import torch
import transformers

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

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(dataset.review,dataset.sentiment,random_state = 0 , 
                                                 stratify = dataset.sentiment)

from tokenizers import BertWordPieceTokenizer
#loading the real tokenizer
tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased',lower = True)
tokenizer.save_pretrained('.')
fast_tokenizer = BertWordPieceTokenizer('vocab.txt',lowercase = True)
fast_tokenizer

def fast_encode(texts, tokenizer, chunk_size=256, maxlen=400):

    tokenizer.enable_truncation(max_length=maxlen)
    tokenizer.enable_padding()
    all_ids = []
    
    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i+chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])
    
    return np.array(all_ids)

X_train = fast_encode(X_train.values, fast_tokenizer, maxlen=400)
X_test = fast_encode(X_test.values, fast_tokenizer, maxlen=400)

def build_model(transformer, max_len=400):
    
    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer(input_word_ids)[0]
    cls_token = sequence_output[:, 0, :]
    out = Dense(1, activation='sigmoid')(cls_token)
    
    model = Model(inputs=input_word_ids, outputs=out)
    model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

bert_model = transformers.BertModel.from_pretrained('distilbert-base-uncased')

model = build_model(bert_model, max_len=400)
model.summary()

history = model.fit(x_train,y_train,batch_size = 32 ,validation_data=(x_test,y_test),epochs = 3)
