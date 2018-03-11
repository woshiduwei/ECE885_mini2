import numpy as np
import pylab as pl

len1 = [0.8184, 0.8545, 0.8484, 0.8572, 0.8530, 0.8468, 0.8494, 0.8402, 0.8432, 0.8443, 0.8489, 0.8356, 0.8442, 0.8428, 0.8380]
len2 = [0.8300, 0.8358, 0.8303, 0.8260, 0.8248, 0.8244, 0.8095, 0.8175, 0.8149, 0.8158, 0.8142, 0.8133, 0.8114, 0.8043, 0.8109]
len = len(len1)
epoch = np.arange(0, len)
import matplotlib.pyplot as plt
plt.figure(1)
plt.plot(epoch, len1)
pl.xlabel('epoch')# make axis labels
pl.ylabel('accuracy')
pl.title('expontential loss learning rate')# give plot a title
plt.figure(2)
plt.plot(epoch, len2)
pl.xlabel('epoch')# make axis labels
pl.ylabel('accuracy')
pl.title('keras default demo')
#plt.show()
hidden_size = 5
Whh = np.random.randn(hidden_size, hidden_size)*0.01 # hidden to hidden
print Whh.shape
a = Whh.shape
print a[0], a[1]