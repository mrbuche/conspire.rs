```math
T_n = \sigma_\mathrm{max}\,g\,\frac{\delta_n}{\delta_n^*}, \qquad
T_t = \sigma_\mathrm{max}\frac{\delta_n^*}{\delta_t^*}\left[g\,\frac{\delta_t}{\delta_t^*} + \frac{g'}{2}\,\frac{\delta_t}{\lambda\,\delta_t^*}\left\langle-\frac{\delta_n}{\delta_n^*}\right\rangle^2\right]
```
where $`g`$ and its derivative $`g'`$ are evaluated at $`\max(\lambda,\lambda_\mathrm{max})`$,
```math
g(\lambda) = \begin{cases}
1/\lambda_1, & \lambda < \lambda_1 \\
1/\lambda, & \lambda_1 \le \lambda < \lambda_2 \\
\dfrac{1-\lambda}{(1-\lambda_2)\,\lambda}, & \lambda_2 \le \lambda < 1 \\
0, & \lambda \ge 1
\end{cases}
```
The second term in $`T_t`$ acts only while loading in compression, and $`g'`$ vanishes while unloading.
