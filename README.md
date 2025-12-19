# Solving PDEs with Neural Networks
Project completed as part of FYS-STK4155 at the University of Oslo during the Autumn 2025.

This repository contains the all the code used to create and examine a neural network used to solve PDE. In particular, we consider the heat equation. 

### Requirements:

The required software packages can be found `requirements.txt` and can be installed using the following terminal command:
```
pip install -r requirements.txt
```

### Repository Structure:

All code is included in the folder `Code/`. This includes both jupyter notebooks `*.ipynb` and python files `*.py`. 

The `*.ipynb` files contain all computation used to produce the figures and data presented in the report `FYS_STK_4155_Project_3.pdf`

The `*.ipynb` files rely on the `.py` files to work.

### The Neural Network implementation

The neural network PDE solver is implemented in python as an abstract base class in `_PDE_NN.py`. This makes it easy to adapt the code for other PDEs by making child classes. The class is initalized by:
```python
_PDE_NN(num_layers,
        layer_size,
        activation,
        optimizer = "adam",
        learning_rate = 0.001,
        reg_param = 0,
        regularizer=None)
```
It has the public methods:
```python
train(self, input_data, epochs=100, print_toggle=False)
```
which trains on the given `input_data`.
```python
test(self, input_data)
```
which returns the loss for the given data. And finally,
```
predict(self, input_data)
```
which predicts on `input_data`

In all cases `input_data` has shape `(batch_size, input_size)`, where batch size is simply the number of input points.

The class also has abstract methods that are required to be defined before the child class can be initialized. These are
```python
_trial_solution(self, x, t, N)
```
which defines the trial solution used in creating solutions. Here `x` and `t` are the two input variables and `N` is the raw output of the neural network.
```python
_pde(self, inputs)
```
defines the PDE that is considered. It is implemented using tensorflow's gradient functionality. In addition, we have two setter functions that set the input and output size of the network:
```python
_set_input_size(self)
_set_output_size(self)
```
For more details see the source code. For an example of a child class see `HeatEqNN.py`

### Forward Euler Scheme Implementation

The forward Euler scheme for the heat equation is implemented as a class in `HeatEqSolver.py`. It is initialized by
```python
HeatEqSolver(initial_state: npt.NDArray[np.float64],
             space_step: float,
             time_step: float,
             duration: float)
```
where `initial_state` needs to be of the a size that agrees with the number of points that result from `space_step` over the interval $[0,1]$.
In addition, it has one public method:
```python
solve(self) -> npt.NDArray[np.float64]
```
which simply solves the PDE and returns an array of solutions.
