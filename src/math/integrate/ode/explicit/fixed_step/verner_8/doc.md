Explicit, twelve-stage, eighth-order, fixed-step, Runge-Kutta method.[^1]

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

[^1]: Fixed-step variant of the [Verner 8](`crate::math::integrate::Verner8`) method.
