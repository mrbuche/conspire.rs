Hyperelastic-viscoplastic solid constitutive models are defined by a Helmholtz free energy density function of the elastic deformation gradient, a plastic dissipation potential, and the Kröner-Lee decomposition of the deformation gradient.

```math
\mathbf{P}:\dot{\mathbf{F}} - \dot{a}(\mathbf{F}_\mathrm{e}) \geq 0
```
The second law of thermodynamics is satisfied by ensuring that the dissipation potential vanishes and is minimized at zero rate,

```math
\frac{\partial\phi}{\partial\mathbf{D}_\mathrm{p}}:\mathbf{D}_\mathrm{p} - \phi \geq 0 \quad\text{and}\quad \phi(\mathbf{0}) = 0
```
by requiring the inequality to hold for arbitrary deformation gradient rates, which yields a relation for the stress,

```math
\mathbf{P} = \frac{\partial a}{\partial\mathbf{F}} \quad\Longrightarrow\quad \mathcal{C}_{iJkL} = \mathcal{C}_{kLiJ}
```
and by minimizing the dissipation potential less the plastic power with respect to the plastic stretching rate, yielding the flow rule.

```math
\mathbf{M}_\mathrm{e}' = \frac{\partial\phi}{\partial\mathbf{D}_\mathrm{p}} \quad\Longleftrightarrow\quad \mathbf{D}_\mathrm{p} = \frac{\partial\phi^*}{\partial\mathbf{M}_\mathrm{e}'}
```
The dissipation potential and its dual sum to the plastic power at the conjugate pair, which therefore must be non-negative.

```math
\phi(\mathbf{D}_\mathrm{p}) + \phi^*(\mathbf{M}_\mathrm{e}') = \mathbf{M}_\mathrm{e}':\mathbf{D}_\mathrm{p} \geq 0
```
