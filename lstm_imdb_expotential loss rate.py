from __future__ import print_function
import keras
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb
from keras.callbacks import LearningRateScheduler
from keras import optimizers
from keras.optimizers import SGD
import numpy as np
max_features = 20000
maxlen = 400  # cut texts after this number of words (among top max_features most common words)
batch_size = 100

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

def step_decay(initial_rate):
   loss, acc = model.evaluate(x_train, y_train, verbose=0)
   lrate = initial_rate * np.exp(loss)
   return lrate

lrate = LearningRateScheduler(step_decay)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_begin(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(initial_rate))


print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128, input_length= maxlen))
model.add(LSTM(128, dropout= 0.2, return_sequences=True))
model.add(LSTM(64, dropout= 0.2, return_sequences=True))
model.add(LSTM(32, dropout = 0.2))
model.add(Dense(1, activation='sigmoid'))
epochs = 15
lrate = 0.01
decay = lrate/epochs
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer = adam,
              metrics=['accuracy'])

print('Train...')
loss, acc = model.evaluate(x_train, y_train, verbose=0)
initial_rate = 0.001 / np.exp(loss)
history = LossHistory()
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test), callbacks=[history])

len = len(history.losses)
epoch = np.arange(1, len)
import matplotlib.pyplot as plt
plt.plot(epoch, len)
plt.show()
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc) #donexx