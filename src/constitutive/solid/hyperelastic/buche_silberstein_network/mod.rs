#[cfg(test)]
mod test;

use crate::{
    constitutive::{
        ConstitutiveError,
        solid::{Solid, TWO_THIRDS, elastic::Elastic, hyperelastic::Hyperelastic},
    },
    math::{
        Current, IDENTITY, Quantity, Rank2, TensorRank2,
        integrate::quadrature::{gauss_laguerre, sphere_product},
        special::extensible_langevin,
    },
    mechanics::{CauchyStress, CauchyTangentStiffness, Deformation, DeformationGradient, Scalar},
    units::{EnergyDensity, Stress},
};
use std::{
    collections::HashMap,
    f64::consts::PI,
    sync::{Arc, LazyLock, RwLock},
};

const NUMBER_OF_LAGUERRE_NODES: usize = 64;
const NUMBER_OF_POLAR_NODES: usize = 20;
const NUMBER_OF_AZIMUTHAL_NODES: usize = 40;

/// Chebyshev fit of `w^{5/2} G_a(w)` in `xi = ln w` over `[LN_W_MIN, LN_W_MAX]`.
/// The kernel is bounded and smooth for all `w > 0` (the chain force is
/// asymptotically linear at both ends), so a modest spectral order matches the
/// live quadrature to full working precision; the stress kernel `G` is derived
/// from this fit and its derivative so the two stay exactly consistent.
const CHEBYSHEV_ORDER: usize = 192;
const LN_W_MIN: Scalar = -9.0;
const LN_W_MAX: Scalar = 13.0;

/// Reference (undeformed) value of the isochoric $`\bar{\mathbf{B}}^{-1}`$.
/// Subtracting the network response evaluated here makes the stress and free
/// energy vanish identically at $`\mathbf{F} = \mathbf{1}`$ rather than to
/// quadrature precision; it drops out elsewhere under the deviatoric operator /
/// as a constant.
const REFERENCE_MATRIX: [[Scalar; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

/// One direction of the sphere quadrature, with the pieces that do not depend
/// on the deformation precomputed.
struct SphereNode {
    direction: [Scalar; 3],
    weight: Scalar,
    /// `u . 1 . u`, computed exactly as [`BucheSilbersteinNetwork::stretch_squared`]
    /// against [`REFERENCE_MATRIX`] so the reference subtraction cancels to the bit.
    reference_stretch: Scalar,
    /// Upper triangle of `u (x) u`: `[xx, yy, zz, xy, xz, yz]`.
    dyad: [Scalar; 6],
}

/// Generalized Gauss-Laguerre ($`\alpha = 1`$): $`\int_0^\infty x\,f(x)\,e^{-x}\,dx \approx \sum_j w_j f(x_j)`$.
static LAGUERRE: LazyLock<(Vec<Scalar>, Vec<Scalar>)> =
    LazyLock::new(|| gauss_laguerre(NUMBER_OF_LAGUERRE_NODES, 1.0));

/// Product rule over the unit sphere, with the per-direction quantities that do
/// not depend on the deformation precomputed.
static SPHERE: LazyLock<Vec<SphereNode>> = LazyLock::new(|| {
    sphere_product(NUMBER_OF_POLAR_NODES, NUMBER_OF_AZIMUTHAL_NODES)
        .into_iter()
        .map(|(direction, weight)| {
            let [x, y, z] = direction;
            SphereNode {
                direction,
                weight,
                reference_stretch: BucheSilbersteinNetwork::stretch_squared(
                    &direction,
                    &REFERENCE_MATRIX,
                ),
                dyad: [x * x, y * y, z * z, x * y, x * z, y * z],
            }
        })
        .collect()
});

/// Per-`(w_0, kappa)` quantities that do not depend on the deformation: the
/// fifth radial moment used for the modulus normalization, and the per-node
/// radial kernels evaluated at $`\mathbf{F} = \mathbf{1}`$ (subtracted per node
/// so the reference cancels to the bit there).
struct ModelConstants {
    fifth_moment: Scalar,
    reference_stress: Vec<Scalar>,
    reference_energy: Vec<Scalar>,
}

type Cache<K, V> = LazyLock<RwLock<HashMap<K, Arc<V>>>>;

static MODEL_CONSTANTS: Cache<(u64, u64), ModelConstants> =
    LazyLock::new(|| RwLock::new(HashMap::new()));

impl ModelConstants {
    fn get(reference_w: Scalar, link_stiffness: Scalar) -> Arc<Self> {
        let key = (reference_w.to_bits(), link_stiffness.to_bits());
        if let Some(constants) = MODEL_CONSTANTS.read().unwrap().get(&key) {
            return constants.clone();
        }
        let kernels = RadialKernels::get(link_stiffness);
        let constants = Arc::new(Self {
            fifth_moment: radial_moment(reference_w, 5.0, |lambda| {
                extensible_langevin::inverse(lambda, link_stiffness)
            }),
            reference_stress: SPHERE
                .iter()
                .map(|node| kernels.radial_stress(reference_w * node.reference_stretch))
                .collect(),
            reference_energy: SPHERE
                .iter()
                .map(|node| kernels.radial_energy(reference_w * node.reference_stretch))
                .collect(),
        });
        MODEL_CONSTANTS
            .write()
            .unwrap()
            .entry(key)
            .or_insert_with(|| constants.clone())
            .clone()
    }
}

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

/// Chebyshev coefficients of `f` on `[a, b]` from `n` Chebyshev-Lobatto samples.
fn chebyshev_coefficients(
    f: impl Fn(Scalar) -> Scalar,
    a: Scalar,
    b: Scalar,
    n: usize,
) -> Vec<Scalar> {
    let samples: Vec<Scalar> = (0..n)
        .map(|k| {
            let x = (PI * k as Scalar / (n - 1) as Scalar).cos();
            f(0.5 * (a + b) + 0.5 * (b - a) * x)
        })
        .collect();
    (0..n)
        .map(|j| {
            let scale = if j == 0 || j == n - 1 { 1.0 } else { 2.0 } / (n - 1) as Scalar;
            scale
                * (0..n)
                    .map(|k| {
                        let edge = if k == 0 || k == n - 1 { 0.5 } else { 1.0 };
                        edge * samples[k]
                            * (PI * j as Scalar * k as Scalar / (n - 1) as Scalar).cos()
                    })
                    .sum::<Scalar>()
        })
        .collect()
}

/// Coefficients of the derivative (with respect to the mapped variable on
/// `[a, b]`) of a Chebyshev series.
fn chebyshev_derivative(coefficients: &[Scalar], a: Scalar, b: Scalar) -> Vec<Scalar> {
    let n = coefficients.len();
    let mut derivative = vec![0.0; n];
    if n >= 2 {
        derivative[n - 2] = 2.0 * (n - 1) as Scalar * coefficients[n - 1];
        for k in (0..n - 2).rev() {
            derivative[k] = derivative[k + 2] + 2.0 * (k + 1) as Scalar * coefficients[k + 1];
        }
        derivative[0] *= 0.5;
    }
    let factor = 2.0 / (b - a);
    derivative.iter_mut().for_each(|d| *d *= factor);
    derivative
}

/// Clenshaw evaluation of a Chebyshev series on `[a, b]`.
fn chebyshev_evaluate(coefficients: &[Scalar], a: Scalar, b: Scalar, x: Scalar) -> Scalar {
    let t = ((2.0 * x - a - b) / (b - a)).clamp(-1.0, 1.0);
    let (two_t, mut d, mut dd) = (2.0 * t, 0.0, 0.0);
    for &c in coefficients.iter().skip(1).rev() {
        (d, dd) = (two_t * d - dd + c, d);
    }
    t * d - dd + coefficients[0]
}

/// Precomputed radial kernels for one link stiffness. Stores the Chebyshev fit
/// of `hat G_a(xi) = w^{5/2} G_a(w)` and its `xi`-derivative; the stress kernel
/// follows from `G(w) = 2 w^{-5/2}[hat G_a - hat G_a']`.
struct RadialKernels {
    energy_fit: Vec<Scalar>,
    energy_fit_derivative: Vec<Scalar>,
}

static RADIAL_KERNELS: Cache<u64, RadialKernels> = LazyLock::new(|| RwLock::new(HashMap::new()));

impl RadialKernels {
    fn get(link_stiffness: Scalar) -> Arc<Self> {
        let key = link_stiffness.to_bits();
        if let Some(kernels) = RADIAL_KERNELS.read().unwrap().get(&key) {
            return kernels.clone();
        }
        let kernels = Arc::new(Self::build(link_stiffness));
        RADIAL_KERNELS
            .write()
            .unwrap()
            .entry(key)
            .or_insert_with(|| kernels.clone())
            .clone()
    }
    fn build(kappa: Scalar) -> Self {
        let energy_fit = chebyshev_coefficients(
            |ln_w| {
                let w = ln_w.exp();
                w.powf(2.5)
                    * radial_moment(w, 2.0, |lambda| {
                        extensible_langevin::helmholtz_free_energy(lambda, kappa)
                    })
            },
            LN_W_MIN,
            LN_W_MAX,
            CHEBYSHEV_ORDER,
        );
        let energy_fit_derivative = chebyshev_derivative(&energy_fit, LN_W_MIN, LN_W_MAX);
        Self {
            energy_fit,
            energy_fit_derivative,
        }
    }
    /// `G_a(w) = int psi*(lambda) lambda^2 e^{-w lambda^2} dlambda`.
    fn radial_energy(&self, w: Scalar) -> Scalar {
        chebyshev_evaluate(&self.energy_fit, LN_W_MIN, LN_W_MAX, w.ln()) / (w * w * w.sqrt())
    }
    /// `G(w) = int eta(lambda) lambda^3 e^{-w lambda^2} dlambda`, from the same
    /// fit as [`Self::radial_energy`] so the two are exactly consistent.
    fn radial_stress(&self, w: Scalar) -> Scalar {
        let ln_w = w.ln();
        let g_a = chebyshev_evaluate(&self.energy_fit, LN_W_MIN, LN_W_MAX, ln_w);
        let g_a_derivative =
            chebyshev_evaluate(&self.energy_fit_derivative, LN_W_MIN, LN_W_MAX, ln_w);
        2.0 * (g_a - g_a_derivative) / (w * w * w.sqrt())
    }
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
    /// limit and drifts above it for finite `number_of_links`. `fifth_moment`
    /// is `int eta(l) l^5 e^{-w_0 l^2} dl`.
    fn raw_shear_modulus(&self, fifth_moment: Scalar) -> Scalar {
        8.0 * PI / 15.0 * self.prefactor() * self.reference_w() * fifth_moment
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
        let kernels = RadialKernels::get(kappa);
        let constants = ModelConstants::get(w_0, kappa);
        let mut network = [0.0; 6];
        for (node, &reference) in SPHERE.iter().zip(&constants.reference_stress) {
            let radial = kernels
                .radial_stress(w_0 * Self::stretch_squared(&node.direction, &matrix))
                - reference;
            let coefficient = node.weight * radial;
            (0..6).for_each(|k| network[k] += coefficient * node.dyad[k]);
        }
        let [xx, yy, zz, xy, xz, yz] = network;
        let network =
            TensorRank2::<3, Current, Current>::from([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]]);
        Ok(network.deviatoric()
            * (self.shear_modulus() * self.prefactor()
                / self.raw_shear_modulus(constants.fifth_moment)
                / jacobian)
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
        let kernels = RadialKernels::get(kappa);
        let constants = ModelConstants::get(w_0, kappa);
        let network: Scalar = SPHERE
            .iter()
            .zip(&constants.reference_energy)
            .map(|(node, &reference)| {
                node.weight
                    * (kernels.radial_energy(w_0 * Self::stretch_squared(&node.direction, &matrix))
                        - reference)
            })
            .sum();
        Ok(
            self.shear_modulus() / self.raw_shear_modulus(constants.fifth_moment)
                * self.prefactor()
                * network
                + 0.5 * self.bulk_modulus() * (0.5 * (jacobian.powi(2) - 1.0) - jacobian.ln()),
        )
    }
}
