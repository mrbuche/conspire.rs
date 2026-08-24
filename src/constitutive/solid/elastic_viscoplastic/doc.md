Elastic-viscoplastic solid constitutive models are defined by an elastic stress tensor function of the elastic deformation gradient, a plastic dissipation potential, and the Kröner-Lee decomposition of the deformation gradient.

```math
\mathbf{P}:\dot{\mathbf{F}} - \mathbf{P}_\mathrm{e}:\dot{\mathbf{F}}_\mathrm{e} \geq 0
```
The second law of thermodynamics is satisfied by ensuring that the dissipation potential is minimized and vanishes at zero rate,

```math
\frac{\partial\phi}{\partial\mathbf{D}_\mathrm{p}}:\mathbf{D}_\mathrm{p} - \phi \geq 0 \quad\text{and}\quad \phi(\mathbf{0}) = 0
```
by requiring the inequality to hold for arbitrary deformation gradient rates, which yields the stress and an asymmetric tangent,

```math
\mathbf{P} = \mathbf{P}_\mathrm{e}\cdot\mathbf{F}_\mathrm{p}^{-T} \quad\Longrightarrow\quad \mathcal{C}_{iJkL} \neq \mathcal{C}_{kLiJ}
```
and by minimizing the dissipation potential less the plastic power with respect to the plastic stretching rate, yielding the flow rule.

```math
\mathbf{M}_\mathrm{e}' = \frac{\partial\phi}{\partial\mathbf{D}_\mathrm{p}} \quad\Longleftrightarrow\quad \mathbf{D}_\mathrm{p} = \frac{\partial\phi^*}{\partial\mathbf{M}_\mathrm{e}'}
```
The dissipation potential and its dual sum to the plastic power at the conjugate pair, which therefore must be non-negative.

```math
\phi(\mathbf{D}_\mathrm{p}) + \phi^*(\mathbf{M}_\mathrm{e}') = \mathbf{M}_\mathrm{e}':\mathbf{D}_\mathrm{p} \geq 0
```
