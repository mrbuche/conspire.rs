#[cfg(test)]
mod test;

use super::{langevin, langevin_derivative, sinhc};
use crate::math::Scalar;
use std::f64::consts::{LN_2, TAU};

/// $`\ln(\sinh\eta / \eta)`$, stable as $`\eta \to \infty`$ where $`\sinh`$
/// itself overflows (there $`\ln\sinh\eta \to \eta - \ln 2`$).
fn ln_sinhc(eta: Scalar) -> Scalar {
    let eta = eta.abs();
    if eta < 20.0 {
        sinhc(eta).ln()
    } else {
        eta - LN_2 - eta.ln()
    }
}

/// Inverse of the reduced extensible freely-jointed chain relation
///
/// ```math
/// y = \mathcal{L}(\eta) + \frac{\eta}{\varkappa},
/// ```
///
/// i.e. the nondimensional single-chain force at nondimensional end-to-end
/// length `y` and nondimensional link stiffness `kappa`. Unlike the bare
/// [`inverse_langevin`](super::inverse_langevin) this has no
/// finite-extensibility singularity, so `y` may exceed 1.
///
/// Cohen's rational inverse Langevin composed with the linear term reduces to a
/// single cubic in the orientational stretch $`u = y - \eta/\varkappa`$,
///
/// ```math
/// (\varkappa + 1)\,u^3 - \varkappa y\,u^2 - (\varkappa + 3)\,u + \varkappa y = 0,
/// ```
///
/// with exactly one root in $`(0, 1)`$; that root (Cardano, trigonometric
/// branch, $`p < 0`$ here) seeds three Newton iterations on the exact relation.
pub fn inverse(nondimensional_end_to_end: Scalar, nondimensional_link_stiffness: Scalar) -> Scalar {
    let (y, kappa) = (nondimensional_end_to_end, nondimensional_link_stiffness);
    let (a, b, c, d) = (kappa + 1.0, -kappa * y, -(kappa + 3.0), kappa * y);
    let shift = b / (3.0 * a);
    let p = (3.0 * a * c - b * b) / (3.0 * a * a);
    let q = (2.0 * b.powi(3) - 9.0 * a * b * c + 27.0 * a * a * d) / (27.0 * a.powi(3));
    let m = 2.0 * (-p / 3.0).sqrt();
    let theta = (3.0 * q / (p * m)).clamp(-1.0, 1.0).acos() / 3.0;
    let mut eta = kappa
        * (y - (0..3)
            .map(|k| m * (theta - TAU * (k as Scalar) / 3.0).cos() - shift)
            .find(|root| (0.0..1.0).contains(root))
            .unwrap_or_else(|| (kappa * y / (kappa + 3.0)).min(1.0 - 1e-12)));
    for _ in 0..3 {
        eta -= (langevin(eta) + eta / kappa - y) / (langevin_derivative(eta) + 1.0 / kappa);
    }
    eta.max(0.0)
}

/// Derivative of [`inverse`] with respect to the nondimensional end-to-end
/// length,
///
/// ```math
/// \frac{d\eta}{dy} = \frac{1}{\mathcal{L}'(\eta) + \varkappa^{-1}}.
/// ```
pub fn inverse_derivative(
    nondimensional_end_to_end: Scalar,
    nondimensional_link_stiffness: Scalar,
) -> Scalar {
    let eta = inverse(nondimensional_end_to_end, nondimensional_link_stiffness);
    1.0 / (langevin_derivative(eta) + 1.0 / nondimensional_link_stiffness)
}

/// Nondimensional isometric Helmholtz free energy per link of the reduced
/// extensible freely-jointed chain, relative to $`y = 0`$:
///
/// ```math
/// \psi^*(y) = y\,\eta - \ln\frac{\sinh\eta}{\eta} - \frac{\eta^2}{2\varkappa},
/// \qquad \eta = \eta(y).
/// ```
///
/// It is the Legendre transform of
/// $`\varphi^*(\eta) = \ln(\sinh\eta/\eta) + \eta^2/2\varkappa`$, so
/// $`d\psi^*/dy = \eta`$; it reduces to the freely-jointed chain result as
/// $`\varkappa \to \infty`$.
pub fn helmholtz_free_energy(
    nondimensional_end_to_end: Scalar,
    nondimensional_link_stiffness: Scalar,
) -> Scalar {
    let (y, kappa) = (nondimensional_end_to_end, nondimensional_link_stiffness);
    let eta = inverse(y, kappa);
    y * eta - ln_sinhc(eta) - eta * eta / (2.0 * kappa)
}
