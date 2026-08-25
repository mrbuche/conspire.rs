The Blatz-Ko hyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The mixing parameter $`f`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The parameter $`n = \kappa/2\mu - 1/3`$ is determined by the bulk and shear moduli.
- The Blatz-Ko model reduces to the rubber case when $`f\to 1`$ and to the foam case when $`f\to 0`$.
- In the tangent stiffness, $`K = (1-f)J^{2n} - fJ^{-2n}`$ and $`K' = 2n\left[(1-f)J^{2n} + fJ^{-2n}\right]`$.

[^1]: P.J. Blatz and W.L. Ko, [Trans. Soc. Rheol. **6**, 223 (1962)](https://doi.org/10.1122/1.548937).
