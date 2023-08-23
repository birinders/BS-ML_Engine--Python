import dense_layer
import metrics


class Model:
    def __init__(self) -> None:
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def remove(self) -> bool:
        if self.layers:
            self.layers.pop()
            return True

        return False

    def layer_count(self) -> int:
        return len(self.layers)

    def show_layers(self) -> None:
        print(self.layers)

    def set_loss_opt(self, *, loss_fxn, optimizer):
        self.loss = loss_fxn
        self.optimizer = optimizer

    def hook_layers(self):
        layer_ct = self.layer_count()
        self.input_layer = dense_layer.LayerInput()

        # Set the input layer
        self.layers[0].prev = self.input_layer
        if layer_ct > 1:
            self.layers[0].next = self.layers[1]

        # Set the middle layers
        for i in range(1, layer_ct - 1):
            self.layers[i].prev = self.layers[i - 1]
            self.layers[i].next = self.layers[i + 1]

        # Set the last layer
        self.layers[-1].prev = self.layers[-2]
        self.layers[-1].next = self.loss

    def finalize(self):
        layer_ct = self.layer_count()
        self.input_layer = dense_layer.LayerInput()
        self.trainable_layers = []

        for i in range(layer_ct):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            elif i < layer_ct - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], "weights"):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

    def forward(self, X):
        # Forward the data through all of the layers

        self.input_layer.forward(X)

        for layer in self.layers:
            layer.forward(layer.prev.output)

        return layer.output

    def backward(self, output, y):
        self.loss.backward(output, y)
        for layer in reversed(self.layers):
            layer.backward(layer.next.d_inputs)

    def train(self, X, y, *, epochs=1, print_every=1):
        for epoch in range(1, epochs + 1):
            # Perform forward pass
            output = self.forward(X)

            # Get losses
            data_loss, reg_loss = self.loss.calculate(output, y)
            net_loss = data_loss + reg_loss

            # Get predictions
            # TODO-
            # Set the actual prediction function later
            predictions = self.output_layer_activation.output

            # Get accuracy
            accuracy_F = metrics.PerformanceMetrics()
            accuracy = accuracy_F.accuracy(predictions, y)

            # Do backward pass
            self.backward(output, y)

            # Optimize
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(
                    f"""epoch: {epoch},
                    acc: {accuracy},
                    loss: {net_loss},
                    data loss: {data_loss},
                    reg_loss: {reg_loss},
                    learning_rate: {self.optimizer.current_learning_rate}
                    """
                )
