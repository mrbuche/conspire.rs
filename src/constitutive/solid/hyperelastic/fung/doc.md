The Fung hyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The extra modulus $`\mu_m`$.
- The exponent $`c`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Fung model reduces to the [Neo-Hookean model](super::NeoHookean) when $`\mu_m\to 0`$ or $`c\to 0`$.

[^1]: Y.C. Fung, [Am. J. Physiol. **213**, 1532 (1967)](https://doi.org/10.1152/ajplegacy.1967.213.6.1532).
