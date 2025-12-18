import numpy as np
import numpy.typing as npt

class HeatEqSolver(object):
    """
    Solve the one-dimensional heat equation using explicit forward-Euler method for a given initial state
    using user specified space and time increments over a user defined duration. Dirichlet boundary conditions
    are assumed (i.e. the temperature at the boundaries is held constant at zero).
    """

    def __init__(self, initial_state: npt.NDArray[np.float64], space_step: float, time_step: float, duration: float):
        """
        Constructor
        """
        self.initial_state: npt.NDArray[np.float64] = initial_state
        self.space_step: float = space_step
        self.time_step: float = time_step
        self.duration: float = duration
        self._num_space_points: int= len(initial_state)
        self._num_time_steps: int = round(duration / time_step)

    def solve(self) -> npt.NDArray[np.float64]:
        """
        Solves the heat equation.

        Returns a 2D numpy array where each row corresponds to the state of the system at a given time step.
        """
        solution = np.zeros((self._num_time_steps+1, self._num_space_points))
        state = self.initial_state
        solution[0,:] = state
        solver_mat = self._init_solver_matrix()
        for time_i in range(1, self._num_time_steps+1):
            state = self._one_step(state, solver_mat)
            solution[time_i,:] = state
        return solution
        

    def _init_solver_matrix(self) -> npt.NDArray[np.float64]:
        """
        Initializes the solver matrix used for iterating the time based on the stored space and time steps.
        """
        r = self.time_step / (self.space_step**2)
        mat_size = self._num_space_points-2
        solver_mat = np.zeros((mat_size, mat_size))

        for space_i in range(0, mat_size):
            solver_mat[space_i, space_i] = 1 - 2 * r
            if space_i != 0:
                solver_mat[space_i, space_i - 1] = r
            if space_i != mat_size-1:
                solver_mat[space_i, space_i + 1] = r
    
        return solver_mat
        
    def _one_step(self, current_state, solver_mat) -> npt.NDArray[np.float64]:
        """
        Does one time step update of the current state using the solver matrix
        """
        current_state[1:-1] = solver_mat @ current_state[1:-1]
        return current_state


