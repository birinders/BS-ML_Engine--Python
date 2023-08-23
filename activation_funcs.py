import numpy as np
import nnfs
from loss import Loss_CategoricalCrossEntropy, Loss

nnfs.init()


class Activation_ReLU:
    def predictions(self, outputs):
        return outputs

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, d_values):
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs):
        # Unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalized values
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

    def backward(self, d_values):
        self.d_inputs = np.empty_like(d_values)

        for index, (single_output, single_dvalues) in enumerate(
            zip(self.output, d_values)
        ):
            single_output = single_output.reshape(-1, 1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(
                single_output, single_output.T
            )

            self.d_inputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Activation_Loss_Softmax(Loss):
    # Creates activation and loss function objects
    def __init__(self):
        self.activation = Activation_Softmax()
        self.loss = Loss_CategoricalCrossEntropy()

    # Forward pass
    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)

        # Set the output
        self.output = self.activation.output

        # Calculate and return loss value
        return self.loss.calculate(self.output, y_true)

    # Backward pass
    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If labels are one-hot encoded,
        # turn them into discrete values
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        # Copy so we can safely modify
        self.d_inputs = dvalues.copy()

        # Calculate gradient
        self.d_inputs[range(samples), y_true] -= 1

        # Normalize gradient
        self.d_inputs = self.d_inputs / samples
