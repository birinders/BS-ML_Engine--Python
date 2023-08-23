import numpy as np
from nnfs.datasets import spiral_data
import nnfs

nnfs.init()


class Layer_Dense:
    def __init__(
        self, n_inputs, n_neurons, l1_weight=0, l1_bias=0, l2_weight=0, l2_bias=0
    ):
        self.weights = 0.01 * np.random.randn(n_neurons, n_inputs).astype(
            dtype=np.float64
        )
        self.biases = np.zeros((1, n_neurons), dtype=np.float64)

        self.l1_weight = l1_weight
        self.l1_bias = l1_bias

        self.l2_weight = l2_weight
        self.l2_bias = l2_bias
        # print(self.weights)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, np.array(self.weights).T) + self.biases

    def backward(self, d_values):
        self.d_weights = np.dot(self.inputs.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)

        if self.l1_weight > 0:
            dl1 = np.ones_like(self.weights)
            dl1[self.weights < 0] = -1
            self.d_weights += self.l1_weight * dl1

        if self.l1_bias > 0:
            dl1 = np.ones_like(self.biases)
            dl1[self.biases < 0] = -1
            self.d_biases += self.l1_bias * dl1

        if self.l2_weight > 0:
            self.d_weights += 2 * self.l2_weight * self.weights.T

        if self.l2_bias > 0:
            self.d_biases += 2 * self.l2_bias * self.biases

        self.d_inputs = np.dot(d_values, self.weights)


class LayerDropout:
    def __init__(self, rate) -> None:
        self.rate = 1 - rate

    def forward(self, inputs):
        self.inputs = inputs
        self.binary_mask = (
            np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        )
        self.output = inputs * self.binary_mask

    def backward(self, d_values):
        self.d_inputs = d_values * self.binary_mask


class LayerInput:
    def forward(self, inputs) -> None:
        self.output = inputs


# X, y = spiral_data(samples=100, classes=3)

# print(X)

# Each layer has only 2 inputs, an X and a Y
# layer1 = Layer_Dense(n_inputs=2, n_neurons=5)
# layer1.forward(X)
# print(layer1.output)
# print(layer1.weights)
# print(layer1.biases)
# print(layer1.output)
