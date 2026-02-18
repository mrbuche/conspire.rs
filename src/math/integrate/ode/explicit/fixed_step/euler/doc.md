Explicit, single-stage, first-order, fixed-step, Runge-Kutta method.[^1]

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
y_{n+1} = y_n + hk_1
```

[^1]: Also known as the Euler method.
