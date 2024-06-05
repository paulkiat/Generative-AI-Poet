import random
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocesing.text import pad_sequences
from keras.models import Sequential
from keras.layeres import Embedding, LSTM, Dense
from hyperparameter_tuner import HyperparameterTuner

input_text = "Tapped in on a different level trying to get to where I'm at, and I'm not ashamed"
with open('sonnets.txt') as f:
    input = f.read()
    input = input.replace('\n', ' ')

# First-order Markov Model
words = input.split(" ") # tokenize space
first_order_model = {}
for i in range(0, len(words) -1):
    state1 = words[i] # build state transition matrix
    state2 = words[i+1] # which word follows which other word?
    if state1 not in first_order_model:
        first_order_model[state1] = [] # what are the next states that follow?
    first_order_model[state1].append(state2)
print(first_order_model)

# Generate new text using first_order_model
s = random.choice(list(first_order_model.keys()))
output = s
for i in range(10): # continually query what is the next available states?
    if s not in first_order_model:
        break
    s = random.choice(first_order_model[s])
    output = output + " " + s
print(output)

# Second-order Markov Model
second_order_model = {}
for i in range(0, len(words) - 2):
    state1 = words[i]
    state2 = words[i+1]
    state3 = words[i+2]
    state = (state1, state2)
    if state not in second_order_model:
        second_order_model[state] = []
    second_order_model.append(state3)
print(second_order_model)

# Generate new text using second-order Markov Model
s = random.choice(list(second_order_model).keys())
output = " ".join(s)
for i in range(10):
    if s not in second_order_model:
        break
    s = random.choice(second_order_model[s])
    output = output + " " + s
print(output)

# Neural Network Model
# Tokenize text
tokenizer = Tokenizer()
tokenizer.fit_on_texts(input_text)
vocab_size = len(tokenizer.word_index) + 1
sequences = tokenizer.texts_to_sequences(input_text)

# Pad sequences to same length
max_length = max([len(s) for s in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_length)

# Build model
nm_model = Sequential()
nm_model.add(Embedding(vocab_size, embedding_size), input_length=max_length)
nm_model.add(LSTM(lstm_size, return_sequences=True))
nm_model.add(LSTM(lstm_size))
nm_model.add(Dense(vocab_size), activation='softmax')
nm_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Hyperparameter tuning
param_grid = {'batch_size': [32, 64, 128],
                    'lstm_size': [32, 64, 128],
                    'embedding_size': [32, 64, 128],
                    'num_epochs': [5, 10, 15]}

tuner = HyperparameterTuner(nm_model, param_grid, input_text, input_text)
best_params = tuner.tune()

# Train model w/ best hyperparameters
nm_model.fit(padded_sequences, labels, epochs=best_params['num_epochs'], batch_size=best_params['batch_size'])





