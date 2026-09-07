The Newtonian viscous fluid constitutive model.

**Parameters**
- The bulk viscosity $`\zeta`$.
- The shear viscosity $`\eta`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.
- The deformation gradient rate $`\dot{\mathbf{F}}`$.

**Internal variables**
- None.

**Notes**
- The rate of deformation is given by $`\mathbf{D}=\tfrac{1}{2}(\mathbf{L}+\mathbf{L}^T)`$ with $`\mathbf{L}=\dot{\mathbf{F}}\cdot\mathbf{F}^{-1}`$.
