Implicit, two-stage, second-order, fixed-step, Runge-Kutta method.[^1]

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
k_2 = f(t_{n+1}, y_{n+1})
```
```math
y_{n+1} = y_n + \frac{h}{2}\left(k_1 + k_2\right)
```

[^1]: Also known as the implicit trapezoidal rule.
