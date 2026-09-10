The Buche-Silberstein hyperelastic solid constitutive model, evaluated as the
full statistical network integral.[^1]

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
- The Cauchy stress and free energy are integrals over the end-to-end vectors
  of the network, split into a one-dimensional radial kernel and an integral
  over the unit sphere (a product rule for now). The single-chain response
  inverts $`\gamma = \mathcal{L}(\eta) + \eta/\varkappa`$
  ([`math::special::extensible_langevin`](crate::math::special::extensible_langevin)).
- The radial kernel $`G_a(w)`$ is precomputed once per link stiffness as a
  Chebyshev fit of $`w^{5/2} G_a(w)`$ in $`\ln w`$; the stress kernel $`G`$
  follows from that fit and its derivative, so the two are exactly consistent.
  The fit matches direct Gauss-Laguerre quadrature to full working precision
  over the Gaussian widths a moderate deformation reaches, degrading toward
  $`\sim 10^{-7}`$ near lock-up for very stiff links.
- The Gaussian equilibrium distribution has width tied to the small-stretch
  force slope, so the ideal-chain / $`N_b \to \infty`$ limit is exact
  incompressible neo-Hookean.
- The effective-chain (Cauchy-Born) reduction of the same physics is
  [`BucheSilberstein`](super::BucheSilberstein).
- The tangent stiffness is not yet implemented (`todo!`).

[^1]: M.R. Buche and M.N. Silberstein, [Phys. Rev. E **102**, 012501 (2020)](https://doi.org/10.1103/PhysRevE.102.012501).
