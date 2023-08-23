import numpy as np
from dense_layer import Layer_Dense


class Loss:
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        return data_loss, self.regularization_loss()

    def regularization_loss_layer(self, layer: Layer_Dense):
        reg_loss = 0

        # for layer in self.trainable_layers:
        # L1 weights and biases
        if layer.l1_weight > 0:
            reg_loss += layer.l1_weight * np.sum(np.abs(layer.weights))
        if layer.l1_bias > 0:
            reg_loss += layer.l1_bias * np.sum(np.abs(layer.biases))

        # L2 weights and biases
        if layer.l2_weight > 0:
            reg_loss += layer.l2_weight * np.sum(np.square(layer.weights))
        if layer.l2_bias > 0:
            reg_loss += layer.l2_bias * np.sum(np.square(layer.biases))

        return reg_loss

    def regularization_loss(self):
        reg_loss = 0

        for layer in self.trainable_layers:
            # L1 weights and biases
            if layer.l1_weight > 0:
                reg_loss += layer.l1_weight * np.sum(np.abs(layer.weights))
            if layer.l1_bias > 0:
                reg_loss += layer.l1_bias * np.sum(np.abs(layer.biases))

            # L2 weights and biases
            if layer.l2_weight > 0:
                reg_loss += layer.l2_weight * np.sum(np.square(layer.weights))
            if layer.l2_bias > 0:
                reg_loss += layer.l2_bias * np.sum(np.square(layer.biases))

        return reg_loss


class Loss_CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        # No of samples in batch
        samples = len(y_pred)

        # Clip data to prevent division by zero,
        # or shifting data mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        ### Probabilities for target values- ###
        # If categorical labels:
        if len(y_true.shape) == 1:
            probabilities = y_pred_clipped[range(samples), y_true]

        # One hot encoded masks:
        elif len(y_true.shape) == 2:
            probabilities = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        negative_log_probs = -np.log(probabilities)
        return negative_log_probs

    def backward(self, d_values, y_true):
        samples = len(d_values)
        labels = len(d_values[0])

        # If labels are sparse, turn them to one-hot vectors
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        # Calculate gradient
        self.d_inputs = -y_true / d_values
        # Normalize gradient
        self.d_inputs = self.d_inputs / samples
