The Buche-Silberstein hyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$.
- The number of links $`N_b`$.
- The nondimensional link stiffness $`\varkappa`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.

**Internal variables**
- None.

**Notes**
- The links are extensible freely-jointed: the single-chain force inverts
  $`\gamma = \mathcal{L}(\eta) + \eta/\varkappa`$, which has no finite-extensibility
  singularity, so no maximum stretch is enforced.
- This is the effective-chain (Cauchy-Born) reduction. The full statistical
  network integral of Ref. [^1] &mdash; the radial/angular split of the stress
  over end-to-end vectors with a Gaussian equilibrium distribution &mdash; is
  not yet implemented here.
- Reduces to the [Arruda-Boyce model](super::ArrudaBoyce) as $`\varkappa\to\infty`$,
  and additionally to the [Neo-Hookean model](super::NeoHookean) as $`N_b\to\infty`$.
- The tangent stiffness is not yet implemented (`todo!`).

[^1]: M.R. Buche and M.N. Silberstein, [Phys. Rev. E **102**, 012501 (2020)](https://doi.org/10.1103/PhysRevE.102.012501).
