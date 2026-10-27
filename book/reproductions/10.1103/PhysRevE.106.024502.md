## Freely jointed chain models with extensible links

Michael R. Buche, Meredith N. Silberstein, Scott J. Grutzik\
[Physical Review E 106 (2), 024502 (2022)](https://doi.org/10.1103/PhysRevE.106.024502)

### Abstract

Analytical relations for the mechanical response of single polymer chains are valuable for modeling purposes, on both the molecular and the continuum scale. These relations can be obtained using statistical thermodynamics and an idealized single-chain model, such as the freely jointed chain model. To include bond stretching, the rigid links in the freely jointed chain model can be made extensible, but this almost always renders the model analytically intractable. Here, an asymptotically correct statistical thermodynamic theory is used to develop analytic approximations for the single-chain mechanical response of this model. The accuracy of these approximations is demonstrated using several link potential energy functions. This approach can be applied to other single-chain models, and to molecular stretching in general.

<figure>

{{#include PhysRevE.106.024502/figure_1.svg}}

<figcaption>

**Figure 1.** The nondimensional single-chain mechanical response $`\gamma(\eta)`$ for the EFJC model, using the full asymptotic (dotted), reduced asymptotic (dashed), and exact (solid) approaches, for varying nondimensional link stiffness $`\kappa\in[10,25,100,1000]`$.

</figcaption>
</figure>

<br>

<!-- langtabs-start -->
```python
{{#include PhysRevE.106.024502/figure_1.py:snippet}}
```
```rust
{{#rustdoc_include PhysRevE.106.024502/figure_1.rs:snippet}}
```

<!-- langtabs-end -->
