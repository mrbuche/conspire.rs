Explicit, second-order, fixed-step method for dynamics.[^1]

```math
\ddot{x} = a(t, x, \dot{x})
```
```math
v_{n+1/2} = v_n + \tfrac{h}{2}\,a_n
```
```math
x_{n+1} = x_n + h\,v_{n+1/2}
```
```math
a_{n+1} = a(t_{n+1}, x_{n+1}, v_{n+1/2})
```
```math
v_{n+1} = v_{n+1/2} + \tfrac{h}{2}\,a_{n+1}
```

It requires one evaluation of the acceleration per step, and is stable for $`h \le 2/\omega_{max}`$. It is symplectic, and conserves a nearby energy, for conservative accelerations and a fixed time step. Accelerations that depend on velocity are evaluated at the half-step velocity, which is no longer symplectic.

[^1]: Also known as the leapfrog or kick-drift-kick method, and equivalent to the central difference method for dynamics.
