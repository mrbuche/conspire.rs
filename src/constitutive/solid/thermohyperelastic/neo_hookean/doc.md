The Neo-Hookean thermohyperelastic solid constitutive model.[^1]

**Parameters**
- The bulk modulus $`\kappa`$.
- The shear modulus $`\mu`$ at the reference temperature.
- The coefficient of thermal expansion $`\alpha`$.
- The reference temperature $`T_\mathrm{ref}`$.

**External variables**
- The deformation gradient $`\mathbf{F}`$.
- The temperature $`T`$.

**Internal variables**
- None.

**Notes**
- The shear modulus is proportional to the temperature, $`\mu(T)=\mu\,T/T_\mathrm{ref}`$, so that the shear response is purely entropic, as in an ideal rubber: its internal energy does not depend on the deformation, stretching it at constant length raises its stress with temperature, and stretching it adiabatically warms it. The thermal expansion enters only through the volumetric response.

[^1]: R.S. Rivlin, [Philos. Trans. R. Soc. London, Ser. A **240**, 459 (1948)](https://doi.org/10.1098/rsta.1948.0002).
