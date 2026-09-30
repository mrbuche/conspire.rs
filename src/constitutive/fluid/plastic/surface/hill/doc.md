The Hill yield surface for orthotropic plasticity.[^1]

**Parameters**
- The coefficient $`F`$.
- The coefficient $`G`$.
- The coefficient $`H`$.
- The coefficient $`L`$.
- The coefficient $`M`$.
- The coefficient $`N`$.

**External variables**
- The deviatoric elastic Mandel stress $`\mathbf{M}_\mathrm{e}'`$.

**Internal variables**
- None.

**Notes**
- The equivalent stress is given by
```math
\phi^2 = F(a_{22}-a_{33})^2 + G(a_{33}-a_{11})^2 + H(a_{11}-a_{22})^2 + 2L\,a_{23}^2 + 2M\,a_{31}^2 + 2N\,a_{12}^2,
\qquad \mathbf{a} = \mathrm{sym}(\mathbf{M}_\mathrm{e}'),
```
in the axes of the intermediate configuration, and the coefficients must be positive.
- The surface is the finite-strain extension, in the deviatoric Mandel stress, of the quadratic criterion of Hill, and the flow is associative.
- The coefficients are normalized so that $`F=G=H=1/3`$ and $`L=M=N=1`$ gives $`\phi=|\mathbf{M}_\mathrm{e}'|`$, reducing the surface to the [von Mises surface](super::VonMises). The yield stress is given by the hardening law it is combined with.

[^1]: R. Hill, [Proc. R. Soc. Lond. A **193**, 281 (1948)](https://doi.org/10.1098/rspa.1948.0045).
