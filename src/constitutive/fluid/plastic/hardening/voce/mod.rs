use super::PlasticHardening;
use crate::{constitutive::ConstitutiveError, math::Quantity, mechanics::Scalar, units::Stress};

#[doc = include_str!("doc.md")]
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
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let decay = (equivalent_plastic_strain * -self.saturation_rate).exp();
        Ok(
            self.yield_stress + self.saturation_stress - self.saturation_stress * decay
                + self.hardening_slope * equivalent_plastic_strain,
        )
    }
    fn hardening_modulus(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        let decay = (equivalent_plastic_strain * -self.saturation_rate).exp();
        Ok(self.hardening_slope + self.saturation_stress * decay * self.saturation_rate)
    }
}
