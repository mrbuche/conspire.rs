The Ogden hyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear moduli $`\mu_n`$ for $`n=1\ldots N`$.
- The exponents $`\alpha_n`$ for $`n=1\ldots N`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Ogden model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N=1`$ and $`\alpha_1\to 2`$.

[^1]: R.W. Ogden, [Proc. R. Soc. Lond. A **326**, 565 (1972)](https://doi.org/10.1098/rspa.1972.0026).
