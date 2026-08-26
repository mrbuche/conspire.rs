The Eulerian Seth-Hill elastic solid constitutive model.[^1]<sup>,</sup>[^2]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The exponent $`m`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Eulerian Seth-Hill strain measure is given by $`\mathbf{e}^{(m)} = \tfrac{1}{m}(\mathbf{V}^m - \mathbf{1})`$.
- The model reduces to the [Hencky model](super::super::hyperelastic::Hencky) when $`m=0`$.
- The model reduces to the [Saint Venant-Kirchhoff model](super::SaintVenantKirchhoff) when $`m=2`$.
- The model reduces to the [Almansi-Hamel model](super::AlmansiHamel) when $`m=-2`$.
- This is the Eulerian counterpart of the [Lagrangian Seth-Hill model](super::SethHillLagrangian).

[^1]: B.R. Seth, *Generalized Strain Measure with Applications to Physical Problems*, AD0266913 (1961).
[^2]: R. Hill, [J. Mech. Phys. Solids **16**, 229 (1968)](https://doi.org/10.1016/0022-5096(68)90031-8).
