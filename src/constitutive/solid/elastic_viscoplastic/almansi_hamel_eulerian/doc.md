The Eulerian Almansi-Hamel elastic-viscoplastic solid constitutive model.

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The initial yield stress $`Y_0`$.
- The isotropic hardening slope $`H`$.
- The rate sensitivity parameter $`m`$.
- The reference flow rate $`d_0`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- The plastic deformation gradient $`\mathbf{F}_\mathrm{p}`$.

**Notes**
- The elastic Almansi-Hamel strain measure is given by $`\mathbf{e}_\mathrm{e}=\tfrac{1}{2}(\mathbf{1}-\mathbf{B}_\mathrm{e}^{-1})`$.
- The elastic response is Cauchy elastic, so no Helmholtz free energy density exists.
