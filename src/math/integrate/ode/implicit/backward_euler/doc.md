Implicit, single-stage, first-order, fixed-step, Runge-Kutta method.[^1]

```math
\frac{dy}{dt} = f(t, y)
```
```math
t_{n+1} = t_n + h
```
```math
k_1 = f(t_{n+1}, y_{n+1})
```
```math
y_{n+1} = y_n + hk_1
```

[^1]: Also known as the backward Euler method.
