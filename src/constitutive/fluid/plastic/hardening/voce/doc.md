The Voce isotropic hardening law.[^1]

**Parameters**
- The initial yield stress $`Y_0`$.
- The linear hardening slope $`H`$.
- The saturation stress $`Q`$.
- The saturation rate $`b`$.

**External variables**
- The equivalent plastic strain $`\varepsilon_\mathrm{p}`$.

**Internal variables**
- None.

**Notes**
- The yield stress is given by
```math
Y(\varepsilon_\mathrm{p}) = Y_0 + H\,\varepsilon_\mathrm{p} + Q\left(1 - e^{-b\,\varepsilon_\mathrm{p}}\right),
```
and the hardening modulus by $`\mathrm{d}Y/\mathrm{d}\varepsilon_\mathrm{p} = H + Qb\,e^{-b\,\varepsilon_\mathrm{p}}`$.
- The linear term vanishes for $`H=0`$, which is the Voce law proper, and the yield stress then saturates at $`Y_0 + Q`$.
- The hardening modulus starts at $`H+Qb`$ and approaches $`H`$ as the plastic strain grows.
- The law is independent of the yield surface it is combined with.

[^1]: E. Voce, J. Inst. Met. **74**, 537 (1948).
