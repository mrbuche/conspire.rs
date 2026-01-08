The Hencky hyperelastic-viscoplastic solid constitutive model.

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
- The Hencky strain measure is given by $`\mathbf{h}=\tfrac{1}{2}\ln(\mathbf{B})`$.
