The Seth-Hill hyperelastic-like solid constitutive model.[^1]<sup>,</sup>[^2]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The exponent $`m`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The generalized (material) Seth-Hill strain is given by $`\mathbf{E}^{(m)} = \frac{1}{m}\left(\mathbf{U}^m - \mathbf{1}\right)`$, with $`\mathbf{U}`$ the right stretch tensor, and reduces to $`\mathbf{E}^{(0)} = \ln\mathbf{U}`$ as $`m\to 0`$.
- The model reduces to the [Hencky model](super::Hencky) when $`m = 0`$, and is not, in general, hyperelastic for other $`m`$.

[^1]: B.R. Seth, in *Second-Order Effects in Elasticity, Plasticity, and Fluid Dynamics*, edited by M. Reiner and D. Abir (Pergamon Press, 1964), pp. 162-172.
[^2]: R. Hill, [Proc. R. Soc. London, Ser. A **314**, 457 (1970)](https://doi.org/10.1098/rspa.1970.0018).
