//! Isotropic hardening laws for plastic and viscoplastic fluid constitutive models.

mod linear;
mod voce;

pub use linear::Linear;
pub use voce::Voce;

use crate::{constitutive::ConstitutiveError, math::Quantity, units::Stress};
use std::fmt::Debug;

/// Required methods for isotropic hardening laws.
///
/// A hardening law gives the yield stress $`Y`$ as a function of the equivalent plastic
/// strain $`\varepsilon_\mathrm{p}`$, and is independent of the shape of the yield
/// surface it is combined with.
pub trait PlasticHardening
where
    Self: Clone + Debug,
{
    /// Returns the initial yield stress.
    fn initial_yield_stress(&self) -> Quantity<Stress>;
    /// Returns the isotropic hardening slope.
    fn hardening_slope(&self) -> Quantity<Stress>;
    /// Calculates and returns the yield stress.
    ///
    /// ```math
    /// Y = Y_0 + H\,\varepsilon_\mathrm{p}
    /// ```
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.initial_yield_stress() + self.hardening_slope() * equivalent_plastic_strain)
    }
    /// Calculates and returns the hardening modulus, the derivative of the yield stress
    /// with respect to the equivalent plastic strain.
    ///
    /// ```math
    /// \frac{\mathrm{d}Y}{\mathrm{d}\varepsilon_\mathrm{p}} = H
    /// ```
    ///
    /// This is the derivative of [`Self::yield_stress`]: a model that overrides one
    /// must override the other, and a wrapper must forward both.
    fn hardening_modulus(
        &self,
        _equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.hardening_slope())
    }
}
