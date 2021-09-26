import numpy as np
from .activators import create_activator

class Layer:
    def __init__(self, in_count, out_count, *, activator_type="linear"):
        self.activator = create_activator(activator_type)
        self.weights   = np.random.normal(0, 1.0/np.sqrt(in_count), (out_count, in_count))
        self.bias      = np.zeros((out_count, ))
        self.d_weights = np.zeros((out_count, in_count))
        self.d_bias    = np.zeros((out_count, ))

    def calc_forward(self, x):
        self.x = x
        p = np.dot(x, self.weights.T) + self.bias
        return self.activator.calc_forward(p)

    def calc_backward(self, dy):
        dz = self.activator.calc_backward(dy)
        dx = np.dot(dz, self.weights)
        dW = np.dot(dz.T, self.x)
        db = dz.sum(axis=0)

        self.d_weights = dW
        self.d_bias    = db

        return dx

    def update(self, learn_rate):
        self.weights -= learn_rate * self.d_weights
        self.bias    -= learn_rate * self.d_bias

class Softmax:
    def calc_forward(self, z):
        exp_z = np.exp(z)
        row_sums = np.sum(exp_z, axis=1)
        self.p = exp_z / row_sums[:, None]
        return np.copy(self.p)

    def calc_backward(self, dp):
        pdp = self.p * dp
        return pdp - self.p * pdp.sum(axis=1, keepdims=True)

    def update(self, learn_rate):
        # Nothing to update
        pass
