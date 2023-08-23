# BS-ML Engine--Python
## Getting Started-
Run ```pip install bs-ml-alpha-0.0.1```<br>
Latest python version recommended for best performance.

- Note- Since this project is still in very early stages of development, there is no comprehensive documentation available yet. Since the API is constantly changing, it is only viable to document the API after the project is out of its ```alpha``` status.

BS-ML Engine is a Machine Learning Engine I've developed from the ground up in Python. This project has truly been a passion project of mine, and is under active development. <br><br>
This project was initiaited for one major goal- to demystify the field of Machine Learning for both me, and the user. That is why, it has been made from the ground up with no external dependencies, other than the very basic [NumPy](https://numpy.org/) library just to accelerate the mathematical computations.

The Engine Currently supports only Classification, since that was the field I was the most interested in.<br>

## Features-
- This Engine currently has 5 Types of Layers- Dense, Activation, Dropout, Loss, and Optimizers.

<h3>0. Creating the Model Object-</h3>

This is a very crucial step in building your ML model, as it streamlines the way data is transferred between layers. You can still build a model by initializing seperate layers to variables. This provides you with a greater degree of freedom, but is not recommended if the user isn't well versed with the API.<br>
- (This form of initializing layers to variables instead of the ```Model``` object will be referred to as the "Headless" method from now on)

```my_model = model.Model() # This builds a Model object as declared in the model.py file```

<h3>1. Dense Layer</h3>

These are the building blocks of our neural network, and is declared in the ```dense_layer.py``` file. This layer provides methods to forward pass the data through the network, as well as backpropagate once the loss is calculated at the output. This layer also acts as the input layer by setting the layer's outputs to the layer's inputs. Dense layers support L1 and L2 regularization as well. <br>
The layer takes in following arguments-<br><br>
```n_inputs, n_neurons, l1_weight=0, l1_bias=0, l2_weight=0, l2_bias=0```

- Note- ```n_inputs``` should equal the number of neurons from the previous layer, or the shape of inputs at the input layer.

#### You can initialize these layers as follows-<br>
- ```my_model.add(dense_layer_mine.Layer_Dense(2, 64)) # A leyer of 2 inputs and 64 neurons```<br>
- Headless- ```dense1 = dense_layer_mine.Layer_Dense(2, 64, l2_weight=5e-4, l2_bias=5e-4)```

<h3>2. Activation Layers</h3>

These are declared under the ```activation_funcs.py``` file. They are usually placed after Dense layers, and act as data threholds before the next layer. They require no additional arguments, and automatically inherit their shapes from the previous layer. There are currently 2 activation functions-<br>
1. ReLU<br>
   Added to the model as follows-<br>
   - ```my_model.add(activation_funcs.Activation_ReLU())```
   - Headless- ```activation1 = activation_funcs.Activation_ReLU()```

2. Softmax<br>
   Added as follows-<br>
   - ```my_model.add(activation_funcs.Activation_Softmax())```
   - Headless- ```activation_softmax = activation_funcs.Activation_Softmax()```

<h3>3. Dropout Layers</h3>

These are declared along with the Dense layer in the ```dense_layer.py``` file. They perform a very crucial step in training, i.e. to prevent overfitting by model growing too reliant on specific neurons. That is, some individual neurons may produce a disproportionately large affect on the outcomes of the model. This might be due to overfitting, and is fixed by disabling random neurons in layers.<br><br>
Our dropout layer takes in a single argument- ```rate```, which is a float between 0 and 1. This is the %age of neurons that will be dropped (their outputs set to 0) for the current layer.

Dropout layers are initialized as-
- ```my_model.add(dense_layer.LayerDropout(0.1)) # Creates a dropout layer which will drop 10% of previous layer's output```
- Headless- ```dropout1 = dense_layer.LayerDropout(0.1)```

<h3>4. Loss Layers</h3>

These are declared under the ```loss.py``` file, and are responsible for calculating a "penalty" for a bad prediction. This is usually the last layer in a neural network, placed just before the optimizers. Lower accuracy yields higher loss, and vice-versa. There's currently only one type of loss layer, which implements the Categorical Cross Entropy method for calculating losses. This is a sister algorithm to the Softmax Activation function, and hence also called the Softmax Loss function. Together form a really efficient combination. Since both of them combine mathematically to solve a single problem, this speeds up this step by over 7x, rather than calculating each individually. (Their combined variation is discussed ahead-)

CCE (Softmax) loss is intialized as-
- ```my_model.set_loss_opt(loss_fxn = loss.Loss_CategoricalCrossEntropy(), optimizer = optimizer)```
- Headless- ```loss_fn = loss.Loss_CategoricalCrossEntropy()```

<h3>5. Optimizers</h3>

Optimizers are declared under the ```optimizers.py``` file, and are responsible for adjusting the Dense Layer's weights and biases according to the selected algorithm once the backward propagation is complete.<br><br>
There are corrently 4 types of optimizers available-
1. Adam
   Adam takes in 5 arguments-<br>
   ```learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999```<br><br>
   1. learning rate- The step size taken while moving towards the minimum value for a loss function.<br>
   2. decay- The amount by which the learning rate decreases at every iteration (The net subtraction decreases inversely to the iteration number)<br>
   3. epsilon- A small value for numerical stability by preventing divide by zero errors.<br>
   4. beta_1- Exponential decay rate for the first moment estimates.
   5. beta_2- Exponential decay rate for the second moment estimates (uncentered variance).

   It is initialized as-
   - ```my_model.set_loss_opt(loss_fxn = loss_fxn, optimizer = optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-5))```
   - Headless- ```optimizer = optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-5)```

2. 












