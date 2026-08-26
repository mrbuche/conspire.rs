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

[^1]: B.R. Seth, [*Generalized Strain Measure with Applications to Physical Problems*](https://archive.org/details/DTIC_AD0266913/page/n3/mode/2up), Mathematics Research Center, University of Wisconsin-Madison, report AD0266913 (1961).
[^2]: R. Hill, [J. Mech. Phys. Solids **16**, 229 (1968)](https://doi.org/10.1016/0022-5096(68)90031-8).
