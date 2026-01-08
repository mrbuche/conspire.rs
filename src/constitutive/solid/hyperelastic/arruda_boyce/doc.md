The Arruda-Boyce hyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The number of links $`N_b`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The nondimensional end-to-end length per link of the chains is $`\gamma=\sqrt{\mathrm{tr}(\mathbf{B}^*)/3N_b}`$.
- The nondimensional force is given by the inverse Langevin function as $`\eta=\mathcal{L}^{-1}(\gamma)`$.
- The initial values are given by $`\gamma_0=\sqrt{1/3N_b}`$ and $`\eta_0=\mathcal{L}^{-1}(\gamma_0)`$.
- The Arruda-Boyce model reduces to the [Neo-Hookean model](super::NeoHookean) when $`N_b\to\infty`$.

[^1]: E.M. Arruda and M.C. Boyce, [J. Mech. Phys. Solids **41**, 389 (1993)](https://doi.org/10.1016/0022-5096(93)90013-6).
