//! Hyperviscoelastic solid constitutive models.
//!
//! ---
//!
//! Hyperviscoelastic solid constitutive models are defined by a Helmholtz free energy density and a viscous dissipation function.
//!
//! ```math
//! \mathbf{P}:\dot{\mathbf{F}} - \dot{a}(\mathbf{F}) - \phi(\mathbf{F},\dot{\mathbf{F}}) \geq 0
//! ```
//! Satisfying the second law of thermodynamics though a minimum viscous dissipation principal yields a relation for the stress.
//!
//! ```math
//! \mathbf{P} = \frac{\partial a}{\partial\mathbf{F}} + \frac{\partial\phi}{\partial\dot{\mathbf{F}}}
//! ```
//! Consequently, the rate tangent stiffness associated with the first Piola-Kirchhoff stress is symmetric for these constitutive models.
//!
//! ```math
//! \mathcal{U}_{iJkL} = \mathcal{U}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod saint_venant_kirchhoff;

pub use saint_venant_kirchhoff::SaintVenantKirchhoff;

use super::{elastic_hyperviscous::ElasticHyperviscous, *};

/// Required methods for hyperviscoelastic solid constitutive models.
pub trait Hyperviscoelastic
where
    Self: ElasticHyperviscous,
{
    /// Calculates and returns the Helmholtz free energy density.
    ///
    /// ```math
    /// a = a(\mathbf{F})
    /// ```
    fn helmholtz_free_energy_density(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError>;
}
