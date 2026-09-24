use super::PlasticHardening;
use crate::{constitutive::ConstitutiveError, math::Quantity, mechanics::Scalar, units::Stress};

/// Voce (saturating) isotropic hardening with a linear term.
///
/// ```math
/// Y(\varepsilon_\mathrm{p}) = Y_0 + H\,\varepsilon_\mathrm{p} + Q\left(1 - e^{-b\,\varepsilon_\mathrm{p}}\right)
/// ```
///
/// The linear term vanishes for $`H = 0`$, which is the Voce law proper.
#[derive(Clone, Debug)]
pub struct Voce {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The linear hardening slope $`H`$, which persists after saturation.
    pub hardening_slope: Quantity<Stress>,
    /// The saturation stress $`Q`$ added to the yield stress at full saturation.
    pub saturation_stress: Quantity<Stress>,
    /// The saturation rate $`b`$.
    pub saturation_rate: Scalar,
}

impl PlasticHardening for Voce {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    /// The initial hardening slope $`H + Qb`$, at zero plastic strain.
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope + self.saturation_stress * self.saturation_rate
    }
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let saturation = 1.0 - (-self.saturation_rate * equivalent_plastic_strain.value()).exp();
        Ok(self.yield_stress
            + self.hardening_slope * equivalent_plastic_strain
            + self.saturation_stress * saturation)
    }
    fn hardening_modulus(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let decay = (-self.saturation_rate * equivalent_plastic_strain.value()).exp();
        Ok(self.hardening_slope + self.saturation_stress * (self.saturation_rate * decay))
    }
}
