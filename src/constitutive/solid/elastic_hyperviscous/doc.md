Elastic-hyperviscous solid constitutive models are defined by an elastic stress tensor function and a viscous dissipation function.

```math
\mathbf{P}:\dot{\mathbf{F}} - \mathbf{P}^e(\mathbf{F}):\dot{\mathbf{F}} \geq 0
```
The second law of thermodynamics is satisfied by ensuring that the viscous dissipation function is convex and vanishes at zero rate,

```math
\frac{\partial\phi}{\partial\dot{\mathbf{F}}}:\dot{\mathbf{F}} - \phi \geq 0 \quad\text{and}\quad \phi(\mathbf{F},\mathbf{0}) = 0
```
and by minimizing the dissipation potential less the stress power with respect to the deformation gradient rate, yielding the stress.

```math
\mathbf{P} = \mathbf{P}^e + \frac{\partial\phi}{\partial\dot{\mathbf{F}}}
```
Consequently, the rate tangent stiffness associated with the first Piola-Kirchhoff stress is symmetric for these constitutive models.

```math
\mathcal{U}_{iJkL} = \mathcal{U}_{kLiJ}
```
