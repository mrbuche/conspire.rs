Explicit, two-stage, second-order, fixed-step, Runge-Kutta method.[^1]

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
k_2 = f(t_n + \tfrac{1}{2} h, y_n + \tfrac{1}{2} h k_1)
```
```math
y_{n+1} = y_n + h k_2
```

[^1]: Also known as the explicit midpoint method.
