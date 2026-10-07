//! Isotropic hardening laws for plastic and viscoplastic fluid constitutive models.

mod linear;
mod voce;

pub use linear::Linear;
pub use voce::Voce;

use crate::{
    constitutive::ConstitutiveError,
    math::Quantity,
    units::{EnergyDensity, Stress},
};
use std::fmt::Debug;

/// Required methods for isotropic hardening laws.
pub trait PlasticHardening
where
    Self: Clone + Debug,
{
    /// Returns the initial yield stress $`Y_0 = Y(0)`$.
    fn initial_yield_stress(&self) -> Quantity<Stress>;
    /// Calculates and returns the yield stress $`Y(\varepsilon_\mathrm{p})`$.
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError>;
    /// Calculates and returns the hardening modulus, the derivative of the yield stress
    /// with respect to the equivalent plastic strain.
    ///
    /// ```math
    /// \frac{\mathrm{d}Y}{\mathrm{d}\varepsilon_\mathrm{p}}
    /// ```
    fn hardening_modulus(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError>;
}

/// Required methods for isotropic hardening laws with a closed-form plastic work.
pub trait PlasticWork
where
    Self: PlasticHardening,
{
    /// Calculates and returns the plastic work density, the integral of the yield stress
    /// over the equivalent plastic strain.
    ///
    /// ```math
    /// w_\mathrm{p}(\varepsilon_\mathrm{p}) = \int_0^{\varepsilon_\mathrm{p}} Y(\varepsilon)\,\mathrm{d}\varepsilon
    /// ```
    fn plastic_work_density(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<EnergyDensity>, ConstitutiveError>;
}
