The Lagrangian Bažant-Itskov elastic solid constitutive model.[^1]<sup>,</sup>[^2]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The exponent $`m`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The Bažant-Itskov strain measure is given by $`\mathbf{h}^{(m)} = \tfrac{1}{2m}(\mathbf{U}^m - \mathbf{U}^{-m})`$.
- The model reduces to the elastic [Hencky model](super::Hencky) when $`m=0`$.
- This is the Lagrangian counterpart of the [Eulerian Bažant-Itskov model](super::BazantItskovEulerian).

[^1]: Z.P. Bažant, [J. Eng. Mater. Technol. **120**, 131 (1998)](https://doi.org/10.1115/1.2807001).
[^2]: M. Itskov, [Mech. Res. Commun. **31**, 507 (2004)](https://doi.org/10.1016/j.mechrescom.2004.02.006).
