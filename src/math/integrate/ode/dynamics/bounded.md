Solves an initial value problem like [`integrate`](Self::integrate), subject to a stability bound.

At each step, `bound` reports the fastest time scale of the system at the current state, the reciprocal of its highest angular frequency. The stability limit is that time scale times the extent of the method's stability region. The time step may use at most the fraction `safety` of that limit, where `safety` is in $`(0, 1]`$. A time step above it is an error.
