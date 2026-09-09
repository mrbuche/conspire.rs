#[cfg(test)]
mod test;

use crate::math::{Scalar, integrate::quadrature::gauss_legendre};
use std::f64::consts::TAU;

/// A direction on the unit sphere and its quadrature weight.
pub type SphereNode = ([Scalar; 3], Scalar);

/// Product quadrature over the unit sphere: Gauss-Legendre in
/// $`\mu = \cos\theta`$ and the midpoint (equispaced) rule in $`\varphi`$.
///
/// Exact for products of a degree $`\le 2 n_\mathrm{polar} - 1`$ polynomial in
/// $`\mu`$ with an azimuthal Fourier mode of order $`< n_\mathrm{azimuthal}/2`$.
/// The weights sum to $`4\pi`$.
///
/// ```math
/// \int_{S^2} f(\mathbf{u})\,d\Omega \approx \sum_i w_i\, f(\mathbf{u}_i)
/// ```
///
/// This is the interim sphere rule; a Lebedev grid is the intended upgrade for
/// anisotropic integrands (it needs many fewer points for the same accuracy).
pub fn sphere_product(n_polar: usize, n_azimuthal: usize) -> Vec<SphereNode> {
    assert!(n_polar > 0 && n_azimuthal > 0);
    let (mu, weights_mu) = gauss_legendre(n_polar);
    let d_phi = TAU / n_azimuthal as Scalar;
    let mut nodes = Vec::with_capacity(n_polar * n_azimuthal);
    for (&m, &w_m) in mu.iter().zip(&weights_mu) {
        let sin_theta = (1.0 - m * m).max(0.0).sqrt();
        for j in 0..n_azimuthal {
            let phi = (j as Scalar + 0.5) * d_phi;
            nodes.push((
                [sin_theta * phi.cos(), sin_theta * phi.sin(), m],
                w_m * d_phi,
            ));
        }
    }
    nodes
}
