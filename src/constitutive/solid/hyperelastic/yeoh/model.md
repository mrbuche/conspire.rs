The Yeoh hyperelastic constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The extra moduli $`\mu_n`$ for $`n=2\ldots N`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Yeoh model reduces to the [Neo-Hookean model](super::NeoHookean) when $`\mu_n\to 0`$ for $`n=2\ldots N`$.

[^1]: O.H. Yeoh, [Rubber Chem. Technol. **66**, 754 (1993)](https://doi.org/10.5254/1.3538343).
