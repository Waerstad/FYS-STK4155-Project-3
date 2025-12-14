import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

class PDE_NN(object):
    """
    Simple feedforward neural network to approximate solutions to the 1D heat equation
    """
    def __init__(self, num_layers, layer_size, activation, optimizer = "adam", learning_rate = 0.001, reg_param = 0, regularizer=None):
        super().__init__()
        policy = tf.keras.mixed_precision.Policy("float64")
        tf.keras.mixed_precision.set_global_policy(policy)

        self.input_size = 2 # number of nodes in input layer
        self.output_size = 1 # number of nodes in output layer

        if regularizer == "L1":
            self.reg = regularizers.l1(reg_param)
        elif regularizer == "L2":
            self.reg = regularizers.l2(reg_param)
        elif regularizer == None:
            self.reg = None
        else:
            raise ValueError("Unsupported regularizer type. Use 'L1', 'L2', or None.")
    
        if optimizer == "adam":
            self.optimizer = keras.optimizers.Adam(learning_rate)
        elif optimizer == "rmsprop":
            self.optimizer = keras.optimizers.RMSprop(learning_rate)
        elif optimizer == "sgd":
            self.optimizer = keras.optimizers.SGD(learning_rate)
        else:
            raise ValueError("Unsupported optimizer type. Use 'adam', 'RMSprop', or 'sgd'.")
        
        # Build the network architecture
        input_layer = layers.Input(shape=(self.input_size,)) # Add input layer
        # add first hidden layer
        hidden_layer = layers.Dense(layer_size, activation=activation,
                                  kernel_regularizer=self.reg,
                                  bias_regularizer=self.reg)(input_layer)
        for _ in range(num_layers-1):
            # Add remaining hidden layers
            layers.Dense(layer_size, activation=activation,
                        kernel_regularizer=self.reg,
                        bias_regularizer=self.reg)(hidden_layer)
        output_layer = layers.Dense(self.output_size, activation='linear')(hidden_layer) # Output layer
        self._model = keras.models.Model(input_layer, output_layer)
    
    def __call__(self, inputs):
        input_data = input_data.astype('float64')
        return self._model(inputs)
    
    def _loss(self, inputs):
        """
        Custom loss function for solving the heat equation.
        """
        x, t = tf.split(inputs, num_or_size_splits=2, axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, t])
            raw_output = self._model(tf.concat([x, t], axis=1))
            u = self._trial_solution(x, t, raw_output)
        
            u_x = tape.gradient(u, x)
            u_t = tape.gradient(u, t)

            u_xx = tape.gradient(u_x, x)
        del tape 
        pde = u_t - u_xx
        loss = tf.reduce_mean(tf.square(pde)) 
        return loss
    
    def _trial_solution(self, x, t, N):
        """
        Trial solution that satisfies initial and boundary conditions.
        """
        # Example trial solution: u(x,t) = (1-t)*I(x) + x*(1 - x)*t*N(x)
        # where I(x) is the initial condition and N(x) is the neural net output
        I = tf.sin(np.pi * x)  # Example initial condition
        return (1-t)*I + x*(1-x)*t*N 
    
    @tf.function
    def _train_step(self, input_data):
        with tf.GradientTape() as tape:
            loss = self._loss(input_data)
            gradients = tape.gradient(loss, self._model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self._model.trainable_variables))
        return loss

    def train(self, input_data, epochs=100):
        """
        Train the neural network on the provided data.
        """
        input_data = input_data.astype('float64')
        for i in range(epochs):
            loss = self._train_step(input_data)
            print(f'Epoch {i+1}, Loss: {loss.numpy()}')

    
    def test(self, input_data):
        """
        Runs an evaluation of the model on provided test data.
        """  # Dummy target data as loss does not depend on it
        input_data = input_data.astype('float64')
        test_loss = self._loss(input_data)
        return test_loss
    
    def predict(self, input_data):
        """
        Predict output for the given input data.
        """
        input_data = input_data.astype('float64')
        network_output = self._model(input_data)
        x, t = tf.split(input_data, num_or_size_splits=2, axis=1)
        return self._trial_solution(x, t, network_output)