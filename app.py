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

# sentences = []
# next_char = []

# for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
#     sentences.append(text[i:i+SEQ_LENGTH])
#     next_char.append(text[i+SEQ_LENGTH])

# x = np.zeros((len(sentences), SEQ_LENGTH, len(charecter)))
# y = np.zeros((len(sentences), len(charecter)))

# for i, sentence in enumerate(sentences):
#     for t, charec in enumerate(sentence):
#         x[i, t, char_to_index[charec]] = 1
#     y[i, char_to_index[next_char[i]]] = 1


models = tf.keras.models.load_model("Text_generator.keras")

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds/ np.sum(exp_preds)
    probas = np.random.multinomial (1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    start_index = random.randint(0, len(text) -SEQ_LENGTH - 1)
    generated = ""
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated += sentence
    print(sentence, end = "")
    for i in range(length):
        x = np.zeros((1, SEQ_LENGTH, len(charecter)))
        for t, character in enumerate(sentence):
            x[0, t, char_to_index [character]] = 1

        predictions = models.predict(x, verbose = 0)[0]
        next_index = sample(predictions, temperature)
        next_char = index_to_char[next_index]

        print(next_char, end = "")
        generated += next_char
        sentence = sentence[1:] + next_char
    
    print("",end = "\n")
    return generated

print("--------------0.2----------------")
print(generate_text(800, 0.2))
print("--------------0.6----------------")
print(generate_text(800, 0.6))
print("--------------0.8----------------")
print(generate_text(800, 0.8))
print("--------------1.0----------------")
print(generate_text(800, 1))