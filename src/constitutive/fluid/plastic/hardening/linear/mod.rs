use super::PlasticHardening;
use crate::{math::Quantity, units::Stress};

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
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}
