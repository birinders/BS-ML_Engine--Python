# BS-ML Engine--Python
## Getting Started-
- Run ```pip install bs-ml-alpha-0.0.1```<br>
(Latest python version recommended for best performance.)

-----------

- Note- Since this project is still in very early stages of development, the API is constantly maturing and evolving. Hence, there is no comprehensive documentation available yet.<br><br>
Since the API is constantly changing, it is only viable to document it once the project is out of its ```alpha``` status.

----------

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
<ol>
<li><strong>ReLU</strong><br>
   Added to the model as follows-<br>
   <ul>
   <li><code>my_model.add(activation_funcs.Activation_ReLU())</code></li>
   <li>Headless- <code>activation1 = activation_funcs.Activation_ReLU()</code></li>
   </ul>
<br><br>
<li><strong>Softmax</strong><br>
   Added as follows-<br>
   <ul>
   <li><code>my_model.add(activation_funcs.Activation_Softmax())</code>
   <li>Headless- <code>activation_softmax = activation_funcs.Activation_Softmax()</code>
   </li>
</ol>

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
<ol>
<li><strong>Adam</strong><br>
   Adam optimizer takes in 5 arguments-<br>
   <code>learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999</code><br><br>
   <ol>
   <li>learning rate- The step size taken while moving towards the minimum value for a loss function.<br>
   <li>decay- The amount by which the learning rate decreases at every iteration (The net subtraction decreases inversely to the iteration number)<br>
   <li>epsilon- A small value to prevent divide by zero errors.<br>
   <li>beta_1- Exponential decay rate for the first moment estimates.
   <li>eta_2- Exponential decay rate for the second moment estimates (uncentered variance).
   </ol>

   It is initialized as-
   <ul>
   <li><code>my_model.set_loss_opt(loss_fxn = loss_fxn, optimizer = optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-5))</code>
   <li>Headless- <code>optimizer = optimizers.Optimizer_Adam(learning_rate=0.05, decay=5e-5)</code>
   </ul>
   
<br>
<li><strong>AdaGrad</strong><br>
   The AdaGrad optimizer takes in 3 arguments-<br>
   <code>learning_rate=0.001, decay=0.0, epsilon=1e-7</code><br><br>
   <ol>
   <li>learning rate- The step size taken while moving towards the minimum value for a loss function.<br>
   <li>decay- The amount by which the learning rate decreases at every iteration (The net subtraction decreases inversely to the iteration number)<br>
   <li>epsilon- A small value to prevent divide by zero errors.<br>
   </ol>
   
   It is initialized as-
   <ul>
   <li><code>my_model.set_loss_opt(loss_fxn = loss_fxn, optimizer = optimizers.Optimizer_AdaGrad(learning_rate=0.01, decay=5e-5))</code>
   <li>Headless- <code>optimizer = optimizers.Optimizer_AdaGrad(learning_rate=0.05, decay=5e-5)</code>
   </ul>

<br>
<li><strong>SGD (Stochaistic Gradient Descent)</strong><br>
   The AdaGrad optimizer takes in 3 arguments-<br>
   <code>learning_rate=1.0, decay=0.0, momentum=0.0</code><br><br>
   <ol>
   <li>learning rate- The step size taken while moving towards the minimum value for a loss function.<br>
   <li>decay- The amount by which the learning rate decreases at every iteration (The net subtraction decreases inversely to the iteration number)<br>
   <li>momentum- Momentum while moving towards a local minima to assist in attaining the global minima.<br>
   </ol>
   
   It is initialized as-
   <ul>
   <li><code>my_model.set_loss_opt(loss_fxn = loss_fxn, optimizer = optimizers.Optimizer_SGD(learning_rate=0.5, decay=1e-3, momentum = 0.5))</code>
   <li>Headless- <code>optimizer = optimizers.Optimizer_SGD(learning_rate=0.5, decay=1e-3, momentum = 0.5)</code>
   </ul>

<br>
<li><strong>RMSProp</strong><br>
   The RMSProp optimizer takes in 4 arguments-<br>
   <code>learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9</code><br><br>
   <ol>
   <li>learning rate- The step size taken while moving towards the minimum value for a loss function.<br>
   <li>decay- The amount by which the learning rate decreases at every iteration (The net subtraction decreases inversely to the iteration number)<br>
   <li>epsilon- A small value to prevent divide by zero errors.<br>
   <li>rho- Controls the moving average decay rate of the squared gradients.<br>
   </ol>
   
   It is initialized as-
   <ul>
   <li><code>my_model.set_loss_opt(loss_fxn = loss_fxn, optimizer = optimizers.Optimizer_RMSProp(learning_rate=0.5, decay=1e-3))</code>
   <li>Headless- <code>optimizer = optimizers.Optimizer_SGD(learning_rate=0.5, decay=1e-3)</code>
   </ul>

</ol>

<h3>6. Special Layers-</h3>
<ol>
<li><strong>Softmax Activation + CCE Loss</strong><br>
   The Softmax activation function and the Categorical Cross Entropy loss functions are 2 parts of the same equation. Hence, they can be combined into a single step for a nearly 7x more efficient combination, as they both assist in solving each other. This combined layer is declared in the <code>activation_funcs.py</code> file, and can be initiated as follows-<br><br>

   <ul>
   <li>Headless- <code>loss_activation = activation_funcs.Activation_Loss_Softmax()</code><br>
   <li> The way to include both the final activation function and the loss function in a single step in the Model object is under development.<br>
   It is scheduled for release in the next update.
   </ul>
   
</ol>

<h3>7. Finalizing and training the Model object-</h3>

You can call the ```finalize()``` method on the Model object to finalize it for training. This step will hook all the layers and chain them in a doubly linked list, so you can efficiently transfer data between adjacent layers by referencing them as ```layer.next``` or ```layer.prev```. This step also distinguishes the trainable layers (dense layers) from the non trainable layers (activation, dropout, loss) to pass them to the optimizers.

- Note- There is also a ```hook_layers()``` method, which is a subset of the ```finalize()``` method. The latter has functionally completely replaced ```hook_layers()```, and the old method will be deprecated in a future update.

Finally, you can call the ```train``` method and pass it the following arguments to start the training process-<br>
```X:input_data, y:input_labels, epochs=1, print_every=1```.<br>
- Note- ```epochs``` and ```print_every``` are madatory keyword only arguments, and cannot be passed positionally.

<h3>7.1 Checking model accuracy-</h3>

The ```metrics.py``` file contains the necessary functions required to calculate accuracy of data.<br>
It can be set up as follows-<br>

<ol>
<li>Create a <code>accuracy</code> object as follows-<br>
<code>accuracy_F = metrics.PerformanceMetrics()</code><br><br>
<li>Call the <code>accuracy</code> method on this object by passing in <code>y_pred, y_true</code> as follows-<br>
<code>acc = accuracy_F.accuracy(y_pred, y_true)</code>
</ol>

The accuracy function automatically accounts for both dense and one-hot encoded predictions.
