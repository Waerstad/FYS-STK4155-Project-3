import numpy as np

## For comparison, define the analytical solution
def analytic_sol(point):
    x,t = point
    return np.exp(-np.pi**2*t)*np.sin(np.pi*x)