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
- The nondimensional Helmholtz free energy of a chain is $`\beta\psi(\gamma)`$.
- The nondimensional end-to-end length per link of a chain is $`\gamma=\sqrt{\mathrm{tr}(\mathbf{B}^*)/3N_b}`$.
- The nondimensional force is given by the inverse Langevin function as $`\eta=\mathcal{L}^{-1}(\gamma)`$.
- The initial values are given by $`\gamma_0=\sqrt{1/3N_b}`$ and $`\eta_0=\mathcal{L}^{-1}(\gamma_0)`$.
- The eight-chain model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N_b\to\infty`$.
