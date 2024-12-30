//! Hyperelastic constitutive models.
//!
//! ---
//!
//! Hyperelastic constitutive models are completely defined by a Helmholtz free energy density function of the deformation gradient.
//!
//! ```math
//! \mathbf{P}:\dot{\mathbf{F}} - \dot{a}(\mathbf{F}) \geq 0
//! ```
//! Satisfying the second law of thermodynamics (here, equivalent to extremized or zero dissipation) yields a relation for the stress.
//!
//! ```math
//! \mathbf{P} = \frac{\partial a}{\partial\mathbf{F}}
//! ```
//! Consequently, the tangent stiffness associated with the first Piola-Kirchoff stress is symmetric for hyperelastic constitutive models.
//!
//! ```math
//! \mathcal{C}_{iJkL} = \mathcal{C}_{kLiJ}
//! ```

#[cfg(test)]
pub mod test;

mod arruda_boyce;
mod fung;
mod gent;
mod mooney_rivlin;
mod neo_hookean;
mod saint_venant_kirchoff;
mod yeoh;

pub use self::{
    arruda_boyce::ArrudaBoyce, fung::Fung, gent::Gent, mooney_rivlin::MooneyRivlin,
    neo_hookean::NeoHookean, saint_venant_kirchoff::SaintVenantKirchoff, yeoh::Yeoh,
};
use super::{elastic::Elastic, *};
use std::fmt::Debug;

/// Required methods for hyperelastic constitutive models.
pub trait Hyperelastic<'a>
where
    Self: Elastic<'a> + Debug,
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
