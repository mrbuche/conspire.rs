The Seth-Hill elastic solid constitutive model.[^1]<sup>,</sup>[^2]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The exponent $`m`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Seth-Hill strain measure is given by $`\mathbf{E}^{(m)} = \tfrac{1}{m}(\mathbf{U}^m - \mathbf{1})`$.
- The model reduces to the [Hencky model](super::Hencky) when $`m=0`$.

[^1]: B.R. Seth, in *Second-Order Effects in Elasticity, Plasticity, and Fluid Dynamics*, edited by M. Reiner and D. Abir (Pergamon Press, 1964), pp. 162-172.
[^2]: R. Hill, [Proc. R. Soc. London, Ser. A **314**, 457 (1970)](https://doi.org/10.1098/rspa.1970.0018).
