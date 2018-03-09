
import numpy as np
history = [1, 2, 3, 4]
len = len(history)
epoch = np.arange(0, len)
print epoch
import matplotlib.pyplot as plt
plt.plot(epoch, history)
plt.show()