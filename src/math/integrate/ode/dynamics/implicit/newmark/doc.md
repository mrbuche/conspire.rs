Implicit, second-order, fixed-step method for dynamics.[^1]

```math
x_{n+1} = x_n + h\,v_n + h^2\left[\left(\tfrac{1}{2} - \beta\right)a_n + \beta\,a_{n+1}\right]
```
```math
v_{n+1} = v_n + h\left[(1 - \gamma)\,a_n + \gamma\,a_{n+1}\right]
```

The new position is the root of the residual of the equations of motion, with the new velocity and acceleration eliminated through the two relations above. The default parameters $`\beta = 1/4`$ and $`\gamma = 1/2`$ are the average acceleration method, which is unconditionally stable and conserves energy for linear problems. The effective tangent is $`\frac{1}{\beta h^2}M + \frac{\gamma}{\beta h}C + K`$.

[^1]: Newmark, N. M. (1959). A method of computation for structural dynamics. *Journal of the Engineering Mechanics Division*, 85(3), 67-94.
