The eight-chain hyperelastic solid constitutive model.

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The single-chain model.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The nondimensional end-to-end length per link of a chain is $`\gamma=\sqrt{\mathrm{tr}(\mathbf{B}^*)/3N_b}`$.
- The nondimensional force is given by the single-chain model as $`\eta=\eta(\gamma)`$.
- The nondimensional stiffness is given by the single-chain model as $`\eta'=\mathrm{d}\eta/\mathrm{d}\gamma`$.
- The initial values are given by $`\gamma_0=\sqrt{1/3N_b}`$ and $`\eta_0=\eta(\gamma_0)`$.
- The eight-chain model reduces to the [Arruda-Boyce model](super::ArrudaBoyce) when $`\eta=\mathcal{L}^{-1}(\gamma)`$.
