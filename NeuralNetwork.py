import numpy as np


class NeuralNetwork:
    def __init__(self):
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Weights
        self.W1 = np.random.randn(self.input_layer_size, self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, self.output_layer_size)

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_prime(z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def cost_function(self, X, y):
        # Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5 * sum((y - self.yHat) ** 2)
        return J

    def cost_function_prime(self, X, y):
        self.a3 = self.forward(X)
        delta3 = np.multiply(-(y - self.a3), self.sigmoid_prime(self.z3))
        grad2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid_prime(self.z2)
        grad1 = np.dot(X.T, delta2)

        return grad1, grad2

    def compute_gradient(self, X, y):
        g1, g2 = self.cost_function_prime(X, y)
        return np.concatenate((g1.ravel(), g2.ravel()))

    def get_params(self):
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def set_params(self, params):
        W1_end = self.input_layer_size * self.hidden_layer_size
        self.W1 = np.reshape(params[0:W1_end], (self.input_layer_size, self.hidden_layer_size))
        W2_end = W1_end + self.hidden_layer_size * self.output_layer_size
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hidden_layer_size, self.output_layer_size))


def compute_numerical_gradient(N, X, y):
    initial_params = N.get_params()
    num_grad = np.zeros(initial_params.shape)
    perturb = np.zeros(initial_params.shape)
    e = 1e-4

    for p in range(len(initial_params)):
        perturb[p] = e
        N.set_params(initial_params + perturb)
        loss2 = N.cost_function(X, y)

        N.set_params(initial_params - perturb)
        loss1 = N.cost_function(X, y)

        num_grad[p] = (loss2 - loss1) / (2 * e)

        perturb[p] = 0

    N.set_params(initial_params)

    return num_grad
