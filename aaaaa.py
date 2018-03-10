import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7, 7)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.optimizers import sgd
from keras.callbacks import LearningRateScheduler
from keras import optimizers
nb_classes = 10

seed = 1337
np.random.seed(seed)
#fix seed for reproductivity
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("x_train original shape", x_train.shape)
print("y_train original shape", y_train.shape)
np.random.seed(1337)
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

def step_decay(initial_rate):
   loss, acc = model.evaluate(X_train, Y_train, verbose=0)
   lrate = initial_rate * np.exp(loss)
   return lrate

lrate = LearningRateScheduler(step_decay)

class LRHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lr = []
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.lr.append(step_decay(initial_rate))



epochs = 50
learning_rate = 0.1
decay_rate = 5e-6
momentum = 0.9

X_train = x_train.reshape(60000, 784)
X_test = x_test.reshape(10000, 784)


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)


input_num_units = 784
hidden1_num_units = 512
hidden2_num_units = 512
hidden3_num_units = 512
hidden4_num_units = 512
hidden5_num_units = 512
output_num_units = 10

epochs = 25
batch_size = 128

model = Sequential([
 Dense(output_dim=hidden1_num_units, input_dim=input_num_units, activation='relu'),
 Dropout(0.2),
 Dense(output_dim=output_num_units, input_dim=hidden1_num_units, activation='softmax'),
 ])

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
loss, acc = model.evaluate(X_train, Y_train, verbose=0)
initial_rate = 0.001 / np.exp(loss)
history = LossHistory()
loss, acc = model.evaluate(X_train, Y_train, verbose=0)
history = LossHistory()
model.fit(X_train, Y_train, batch_size=128, epochs=4, verbose=0, callbacks=[history], shuffle= True)
len = len(history.losses)
epoch = np.arange(0, len)
for i in range(len):
    print ('epoch： %d, loss: %.4f' % (i + 1, history.losses[i]))
import matplotlib.pyplot as plt
plt.plot(epoch, history.losses)
plt.show()

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', loss)
print('Test acc:', acc)

