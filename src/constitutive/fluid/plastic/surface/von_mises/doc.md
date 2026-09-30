The von Mises yield surface.[^1]

**Parameters**
- None.

**External variables**
- The deviatoric elastic Mandel stress $`\mathbf{M}_\mathrm{e}'`$.

**Internal variables**
- None.

**Notes**
- The equivalent stress is given by
```math
\phi(\mathbf{M}_\mathrm{e}') = |\mathbf{M}_\mathrm{e}'|,
```
the Frobenius norm of the deviator, and the flow is associative, with the flow direction $`\mathbf{N} = \mathbf{M}_\mathrm{e}'/|\mathbf{M}_\mathrm{e}'|`$.
- The equivalent stress is not normalized to the uniaxial stress: in uniaxial tension it is $`\sqrt{2/3}`$ times the axial stress, so the yield stress of a hardening law is $`\sqrt{2/3}`$ times the uniaxial yield stress.
- This is the isotropic case of the [Hill surface](super::Hill), with $`F=G=H=1/3`$ and $`L=M=N=1`$.

[^1]: R. von Mises, Mechanik der festen Körper im plastisch-deformablen Zustand, *Nachr. Ges. Wiss. Göttingen, Math.-Phys. Kl.*, **4**, 582 (1913).
