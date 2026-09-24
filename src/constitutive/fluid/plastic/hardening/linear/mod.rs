use super::PlasticHardening;
use crate::{constitutive::ConstitutiveError, math::Quantity, units::Stress};

/// Linear isotropic hardening.
///
/// ```math
/// Y(\varepsilon_\mathrm{p}) = Y_0 + H\,\varepsilon_\mathrm{p}
/// ```
#[derive(Clone, Debug)]
pub struct Linear {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Quantity<Stress>,
}

impl PlasticHardening for Linear {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.yield_stress + self.hardening_slope * equivalent_plastic_strain)
    }
    fn hardening_modulus(
        &self,
        _equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.hardening_slope)
    }
}
