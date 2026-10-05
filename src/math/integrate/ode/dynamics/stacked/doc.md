Marks an explicit integrator for ordinary differential equations as usable for dynamics, by integrating the position and velocity together as a single state.

```math
\frac{d}{dt}\begin{bmatrix} x \\ v \end{bmatrix} = \begin{bmatrix} v \\ a(t, x, v) \end{bmatrix}
```

Every integrator that is [`Stacked`] is also an [`ExplicitDynamics`](crate::math::integrate::ExplicitDynamics) integrator. It is implemented only for methods whose stability region reaches the imaginary axis, which excludes the first- and second-order methods, since those are unstable for undamped oscillation.

These methods are not symplectic, and evaluate the acceleration once per stage rather than once per step.
