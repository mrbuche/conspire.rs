//! Hyperviscoelastic constitutive models.
//!
//! ---
//!
//! Hyperviscoelastic constitutive models are defined by a Helmholtz free energy density function and a viscous dissipation function.
//!
//! ```math
//! \mathbf{P}:\dot{\mathbf{F}} - \dot{a}(\mathbf{F}) - \phi(\mathbf{F},\dot{\mathbf{F}}) \geq 0
//! ```
//! Satisfying the second law of thermodynamics though a minimum viscous dissipation principal yields a relation for the stress.
//!
//! ```math
//! \mathbf{P} = \frac{\partial a}{\partial\mathbf{F}} + \frac{\partial\phi}{\partial\dot{\mathbf{F}}}
//! ```
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchoff stress is symmetric for hyperviscoelastic models.
//!
//! ```math
//! \mathcal{U}_{iJkL} = \mathcal{U}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod saint_venant_kirchoff;

pub use saint_venant_kirchoff::SaintVenantKirchoff;

use super::{
    super::fluid::viscous::Viscous, elastic_hyperviscous::ElasticHyperviscous,
    viscoelastic::Viscoelastic, *,
};
use std::fmt::Debug;

/// Required methods for hyperviscoelastic constitutive models.
pub trait Hyperviscoelastic<'a>
where
    Self: ElasticHyperviscous<'a>,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn calculate_helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError>;
}
