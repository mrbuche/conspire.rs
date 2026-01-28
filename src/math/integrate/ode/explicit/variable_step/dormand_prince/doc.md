Explicit, six-stage, fifth-order, variable-step, Runge-Kutta method.[^1]

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
k_2 = f(t_n + \tfrac{1}{5} h, y_n + \tfrac{1}{5} h k_1)
```
```math
k_3 = f(t_n + \tfrac{3}{10} h, y_n + \tfrac{3}{40} h k_1 + \tfrac{9}{40} h k_2)
```
```math
k_4 = f(t_n + \tfrac{4}{5} h, y_n + \tfrac{44}{45} h k_1 - \tfrac{56}{15} h k_2 + \tfrac{32}{9} h k_3)
```
```math
k_5 = f(t_n + \tfrac{8}{9} h, y_n + \tfrac{19372}{6561} h k_1 - \tfrac{25360}{2187} h k_2 + \tfrac{64448}{6561} h k_3 - \tfrac{212}{729} h k_4)
```
```math
k_6 = f(t_n + h, y_n + \tfrac{9017}{3168} h k_1 - \tfrac{355}{33} h k_2 - \tfrac{46732}{5247} h k_3 + \tfrac{49}{176} h k_4 - \tfrac{5103}{18656} h k_5)
```
```math
y_{n+1} = y_n + h\left(\frac{35}{384}\,k_1 + \frac{500}{1113}\,k_3 + \frac{125}{192}\,k_4 - \frac{2187}{6784}\,k_5 + \frac{11}{84}\,k_6\right)
```
```math
k_7 = f(t_{n+1}, y_{n+1})
```
```math
e_{n+1} = \frac{h}{5}\left(\frac{71}{11520}\,k_1 - \frac{71}{3339}\,k_3 + \frac{71}{384}\,k_4 - \frac{17253}{67840}\,k_5 + \frac{22}{105}\,k_6 - \frac{1}{8}\,k_7\right)
```

[^1]: J.R. Dormand and P.J. Prince, [J. Comput. Appl. Math. **6**, 19 (1980)](https://doi.org/10.1016/0771-050X(80)90013-3).
