import matplotlib.pyplot as plt
import numpy as np
from NeuralNetwork import NeuralNetwork
from Trainer import Trainer

X = np.array(([3, 5], [5, 1], [10, 2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)

# Normalize
X = X / np.amax(X, axis=0)
y = y / 100  # Max test score is 100

NN = NeuralNetwork()
T = Trainer(NN)
T.train(X, y)

plt.plot(T.J)
plt.grid(1)
plt.ylabel('Cost')
plt.xlabel('Iterations')
plt.show()
