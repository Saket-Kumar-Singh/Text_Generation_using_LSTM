import random
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# from tensorflow import keras
from tensorflow.keras.layers import LSTM, Activation, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop 
import tensorflow as tf

filepath = tf.keras.utils.get_file('sheakspear.txt', "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

text = open(filepath, 'rb').read().decode(encoding = "utf-8").lower()

text = text[30000:80000]

charecter = sorted(set(text))

char_to_index = dict((c, i) for i, c in enumerate(charecter))
index_to_char = dict((i, c) for i, c in enumerate(charecter))

SEQ_LENGTH = 200
STEP_SIZE = 3

sentences = []
next_char = []

for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_char.append(text[i+SEQ_LENGTH])

x = np.zeros((len(sentences), SEQ_LENGTH, len(charecter)))
y = np.zeros((len(sentences), len(charecter)))

for i, sentence in enumerate(sentences):
    for t, charec in enumerate(sentence):
        x[i, t, char_to_index[charec]] = 1
    y[i, char_to_index[next_char[i]]] = 1

models = Sequential()
models.add(LSTM(128, input_shape = (SEQ_LENGTH, len(charecter))))
models.add(Dense(len(charecter)))
models.add(Activation('softmax'))

models.compile(loss = "categorical_crossentropy", optimizer = RMSprop(learning_rate = 0.01))

models.fit(x, y, batch_size = 256, epochs = 8)

models.save("Text_generator.keras")