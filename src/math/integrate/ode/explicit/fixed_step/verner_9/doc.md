Explicit, fifteen-stage, ninth-order, fixed-step, Runge-Kutta method.[^1]

```math
\frac{dy}{dt} = f(t, y)
```
```math
t_{n+1} = t_n + h
```
```math
k_1 = f(t_n, y_n)
```
```math
\cdots
```

[^1]: Fixed-step variant of the [Verner 9](`crate::math::integrate::Verner9`) method.
