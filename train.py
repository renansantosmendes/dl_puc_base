import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import numpy as np
import random

import os
import mlflow
from mlflow import MlflowClient

from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

def reset_seeds():
   np.random.seed(123)
   random.seed(123)
   tf.random.set_seed(1234)

(x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=5000)
x_train = pad_sequences(x_train, 500)
x_test = pad_sequences(x_test, 500)

word_to_id = imdb.get_word_index()
word_to_id = {k: (v+3) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {idx: word for word, idx in word_to_id.items()}

MLFLOW_TRACKING_URI = 'https://dagshub.com/renansantosmendes/DL202202L06.mlflow'
MLFLOW_TRACKING_USERNAME = 'renansantosmendes'
MLFLOW_TRACKING_PASSWORD = 'cc41cc48f8e489dd5b87404dd6f9720944e32e9b'

os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

mlflow.tensorflow.autolog(log_models=True,
                          log_input_examples=True,
                          log_model_signatures=True)

reset_seeds()
embedding_vector_length = 100
model = Sequential()

model.add(Embedding(
    input_dim=5000,
    output_dim=embedding_vector_length,
    input_length=500
))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

with mlflow.start_run(run_name='imdb_review'):
  reset_seeds()
  model.fit(x_train,
            y_train,
            validation_data=(x_test, y_test),
            epochs=1,
            batch_size=512)