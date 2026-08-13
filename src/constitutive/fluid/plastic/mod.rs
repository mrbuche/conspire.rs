//! Plastic fluid constitutive models.

use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, unit::Stress},
};
use std::fmt::Debug;

/// Required methods for plastic fluid constitutive models.
pub trait Plastic
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
        //
        // Can eventually make a subdirectory with an enum (like LineaSearch) with different hardening models.
        //
        Ok(self.initial_yield_stress() + self.hardening_slope() * equivalent_plastic_strain)
    }
}
