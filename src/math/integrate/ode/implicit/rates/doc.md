Solves an initial value problem for a first-order system, given the residual of the system as a function of time, state, and rate, and its tangent with respect to the state.

The rate function is only evaluated for the initial rate. The tangent is handed the time step, from which the weight of the capacity in the effective tangent follows. The solution contains the states and rates at each time. The time step is that of the integrator when `time` holds only the initial and final times, and otherwise `time` is the sequence of steps.
