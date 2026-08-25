The Ogden hyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The moduli $`\mu_n`$ for $`n=1\ldots N`$.
- The exponents $`\alpha_n`$ for $`n=1\ldots N`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Ogden model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N=1`$ and $`\alpha_1\to 2`$.
- The shear modulus is given by $`2\mu = \sum_{n=1}^N \mu_n\alpha_n`$.
- $`\mathbf{V}=\mathbf{B}^{1/2}`$ is the left stretch tensor, with $`\mathbf{V}^*=(\mathbf{V}/J^{1/3})`$ its isochoric part; both share the eigenvectors of $`\mathbf{B}=\mathbf{F}\mathbf{F}^T`$, with eigenvalues equal to the principal stretches.

[^1]: R.W. Ogden, [Proc. R. Soc. Lond. A **326**, 565 (1972)](https://doi.org/10.1098/rspa.1972.0026).
