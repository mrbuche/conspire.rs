#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{
        Current, IDENTITY, Quantity, Rank2, TensorArray, TensorRank2,
        integrate::quadrature::{SphereNode, gauss_laguerre, sphere_product},
        special::extensible_langevin,
    },
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};
use std::{f64::consts::PI, sync::LazyLock};

const NUMBER_OF_LAGUERRE_NODES: usize = 32;
const NUMBER_OF_POLAR_NODES: usize = 24;
const NUMBER_OF_AZIMUTHAL_NODES: usize = 48;

/// Reference (undeformed) value of the isochoric $`\bar{\mathbf{B}}^{-1}`$.
/// Subtracting the network response evaluated here makes the stress and free
/// energy vanish identically at $`\mathbf{F} = \mathbf{1}`$ rather than to
/// quadrature precision; it drops out elsewhere under the deviatoric operator /
/// as a constant.
const REFERENCE_MATRIX: [[Scalar; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// Generalized Gauss-Laguerre ($`\alpha = 1`$): $`\int_0^\infty x\,f(x)\,e^{-x}\,dx \approx \sum_j w_j f(x_j)`$.
static LAGUERRE: LazyLock<(Vec<Scalar>, Vec<Scalar>)> =
    LazyLock::new(|| gauss_laguerre(NUMBER_OF_LAGUERRE_NODES, 1.0));
static SPHERE: LazyLock<Vec<SphereNode>> =
    LazyLock::new(|| sphere_product(NUMBER_OF_POLAR_NODES, NUMBER_OF_AZIMUTHAL_NODES));

#[doc = include_str!("doc.md")]
#[derive(Clone, Debug)]
pub struct BucheSilbersteinNetwork {
    /// The bulk modulus $`\kappa`$.
    pub bulk_modulus: Quantity<Stress>,
    /// The shear modulus $`\mu`$.
    pub shear_modulus: Quantity<Stress>,
    /// The number of links $`N_b`$.
    pub number_of_links: Scalar,
    /// The nondimensional link stiffness $`\varkappa`$.
    pub link_stiffness: Scalar,
}

/// $`\int_0^\infty f(\lambda)\,\lambda^m\,e^{-w\lambda^2}\,d\lambda`$ by the
/// substitution $`x = w\lambda^2`$ and the generalized Gauss-Laguerre rule:
///
/// ```math
/// \tfrac{1}{2}\,w^{-(m+1)/2}\sum_j w_j\, x_j^{(m-3)/2}\, f\!\left(\sqrt{x_j/w}\right).
/// ```
fn radial_moment(w: Scalar, m: Scalar, f: impl Fn(Scalar) -> Scalar) -> Scalar {
    let (nodes, weights) = &*LAGUERRE;
    let power = 0.5 * (m - 3.0);
    0.5 * w.powf(-0.5 * (m + 1.0))
        * nodes
            .iter()
            .zip(weights)
            .map(|(&x, &weight)| weight * x.powf(power) * f((x / w).sqrt()))
            .sum::<Scalar>()
}

impl BucheSilbersteinNetwork {
    /// Returns the number of links.
    pub fn number_of_links(&self) -> Scalar {
        self.number_of_links
    }
    /// Returns the nondimensional link stiffness.
    pub fn link_stiffness(&self) -> Scalar {
        self.link_stiffness
    }
    /// Small-stretch force slope $`k_1 = 1/(1/3 + 1/\varkappa)`$; the Gaussian
    /// equilibrium distribution has width $`\propto k_1`$ (distribution-behavior
    /// correspondence) so the ideal-chain limit is exact neo-Hookean.
    fn slope(&self) -> Scalar {
        1.0 / (1.0 / 3.0 + 1.0 / self.link_stiffness())
    }
    /// $`w_0 = k_1 N_b / 2`$, the Gaussian exponent coefficient at unit stretch.
    fn reference_w(&self) -> Scalar {
        0.5 * self.slope() * self.number_of_links()
    }
    /// $`N_b^{5/2} (k_1 / 2\pi)^{3/2}`$.
    fn prefactor(&self) -> Scalar {
        self.number_of_links().powf(2.5) * (self.slope() / (2.0 * PI)).powf(1.5)
    }
    /// The raw (un-normalized) small-stretch shear modulus of the network
    /// integral, in units of the shear modulus; equals 1 in the ideal-chain
    /// limit and drifts above it for finite `number_of_links`.
    fn raw_shear_modulus(&self) -> Scalar {
        let (w_0, kappa) = (self.reference_w(), self.link_stiffness());
        8.0 * PI / 15.0
            * self.prefactor()
            * w_0
            * radial_moment(w_0, 5.0, |lambda| {
                extensible_langevin::inverse(lambda, kappa)
            })
    }
    /// $`\mathbf{u}\cdot\bar{\mathbf{B}}^{-1}\cdot\mathbf{u}`$ for a unit vector.
    fn stretch_squared(
        direction: &[Scalar; 3],
        isochoric_left_cauchy_green_inverse: &[[Scalar; 3]; 3],
    ) -> Scalar {
        (0..3)
            .map(|i| {
                direction[i]
                    * (0..3)
                        .map(|j| isochoric_left_cauchy_green_inverse[i][j] * direction[j])
                        .sum::<Scalar>()
            })
            .sum()
    }
}

impl Solid for BucheSilbersteinNetwork {
    fn bulk_modulus(&self) -> Quantity<Stress> {
        self.bulk_modulus
    }
    fn shear_modulus(&self) -> Quantity<Stress> {
        self.shear_modulus
    }
}

impl Elastic for BucheSilbersteinNetwork {
    #[doc = include_str!("cauchy_stress.md")]
    fn cauchy_stress(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyStress, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_inverse =
            (deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS)).inverse();
        let matrix: [[Scalar; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| isochoric_left_cauchy_green_inverse[i][j].value())
        });
        let (w_0, kappa) = (self.reference_w(), self.link_stiffness());
        let force = |lambda: Scalar| extensible_langevin::inverse(lambda, kappa);
        let mut network = TensorRank2::<3, Current, Current>::zero();
        for (direction, weight) in SPHERE.iter() {
            let radial = radial_moment(w_0 * Self::stretch_squared(direction, &matrix), 3.0, force)
                - radial_moment(
                    w_0 * Self::stretch_squared(direction, &REFERENCE_MATRIX),
                    3.0,
                    force,
                );
            network += TensorRank2::from(std::array::from_fn(|i| {
                std::array::from_fn(|j| direction[i] * direction[j])
            })) * (weight * radial);
        }
        Ok(network.deviatoric()
            * (self.shear_modulus() * self.prefactor() / self.raw_shear_modulus() / jacobian)
            + IDENTITY * self.bulk_modulus() * 0.5 * (jacobian - 1.0 / jacobian))
    }
    #[doc = include_str!("cauchy_tangent_stiffness.md")]
    fn cauchy_tangent_stiffness(
        &self,
        _deformation_gradient: &DeformationGradient,
    ) -> Result<CauchyTangentStiffness, ConstitutiveError> {
        todo!("analytic tangent stiffness for the Buche-Silberstein network model")
    }
}

impl Hyperelastic for BucheSilbersteinNetwork {
    #[doc = include_str!("helmholtz_free_energy_density.md")]
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError> {
        let jacobian = self.jacobian(deformation_gradient)?;
        let isochoric_left_cauchy_green_inverse =
            (deformation_gradient.left_cauchy_green() / jacobian.powf(TWO_THIRDS)).inverse();
        let matrix: [[Scalar; 3]; 3] = std::array::from_fn(|i| {
            std::array::from_fn(|j| isochoric_left_cauchy_green_inverse[i][j].value())
        });
        let (w_0, kappa) = (self.reference_w(), self.link_stiffness());
        let psi = |w: Scalar| {
            radial_moment(w, 2.0, |lambda| {
                extensible_langevin::helmholtz_free_energy(lambda, kappa)
            })
        };
        let network: Scalar = SPHERE
            .iter()
            .map(|(direction, weight)| {
                weight
                    * (psi(w_0 * Self::stretch_squared(direction, &matrix))
                        - psi(w_0 * Self::stretch_squared(direction, &REFERENCE_MATRIX)))
            })
            .sum();
        Ok(
            self.shear_modulus() / self.raw_shear_modulus() * self.prefactor() * network
                + 0.5 * self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()),
        )
    }
}
