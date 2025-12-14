import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

class HeatEqNN(keras.Sequential):
    def __init__(self, num_layers, layer_size, activation, reg_param, regularizer=None):
        super().__init__()
        if regularizer == "L1":
            reg = regularizers.l1(reg_param)
        elif regularizer == "L2":
            reg = regularizers.l2(reg_param)
        elif regularizer != None:
            raise ValueError("Unsupported regularizer type. Use 'L1', 'L2', or None.")
        self.add(layers.InputLayer(input_shape=(2,))) # Add input layer containing (x,y) coordinate of input point.
        for _ in range(num_layers):
            # Add hidden layers
            self.add(layers.Dense(layer_size, activation=activation,
                                  kernel_regularizer=reg,
                                  bias_regularizer=reg))
        self.add(layers.Dense(2, activation='linear')) # Output layer with 2 outputs (x,y)
