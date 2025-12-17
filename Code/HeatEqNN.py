from _PDE_NN import _PDE_NN
import numpy as np
import tensorflow as tf

class HeatEqNN(_PDE_NN):

    def _pde(self, inputs):
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
        return pde
    
    def _trial_solution(self, x, t, N):
        """
        Trial solution that satisfies initial and boundary conditions.
        """
        # Example trial solution: u(x,t) = (1-t)*I(x) + x*(1 - x)*t*N(x)
        # where I(x) is the initial condition and N(x) is the neural net output
        I = tf.sin(np.pi * x)  # Example initial condition
        return (1-t)*I + x*(1-x)*t*N 