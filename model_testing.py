from all import *

X, y = spiral_data(1000, 5)

my_model = model.Model()
my_model.add(dense_layer.Layer_Dense(2, 50))
my_model.add(activation_funcs.Activation_ReLU())
my_model.add(dense_layer.Layer_Dense(50, 5))
my_model.add(activation_funcs.Activation_Softmax())
# my_model.add(dense_layer_mine.Layer_Dense(64, 3))
# my_model.add(activation_funcs.Activation_ReLU())

my_model.set_loss_opt(
    loss_fxn=loss.Loss_CategoricalCrossEntropy(),
    optimizer=optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-5),
)

my_model.finalize()

my_model.train(X, y, epochs=10000, print_every=100)
