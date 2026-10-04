Solves an initial value problem like [`integrate`](Self::integrate), subject to a stability bound.

At each step, `bound` reports the fastest time scale of the right-hand side at the current state. The stability limit is that time scale times the extent of the method's stability region along the reported axis. The time step may use at most the fraction `safety` of that limit, where `safety` is in $`(0, 1]`$. A fixed time step above it is an error, and an adaptive time step is reduced to it.
