//! Viscoplastic fluid constitutive models.

use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::Plastic},
    math::{Scalar, Tensor, TensorArray},
    mechanics::{MandelStressElastic, StretchingRatePlastic},
};

const TWO_THIRDS: Scalar = 2.0 / 3.0;

/// Required methods for viscoplastic fluid constitutive models.
pub trait Viscoplastic
where
    Self: Plastic,
{
    /// Calculates and returns the rate of plastic stretching.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = d_0\left(\frac{|\mathbf{M}_\mathrm{e}'|}{Y(S)}\right)^{\footnotesize\tfrac{1}{m}}\frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress_e: MandelStressElastic,
        yield_stress: Scalar,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress_e.norm();
        if magnitude == 0.0 {
            Ok(StretchingRatePlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress_e
                * (self.reference_flow_rate() / magnitude
                    * (magnitude / yield_stress).powf(1.0 / self.rate_sensitivity())))
        }
    }
    /// Returns the rate_sensitivity parameter.
    fn rate_sensitivity(&self) -> Scalar;
    /// Returns the reference flow rate.
    fn reference_flow_rate(&self) -> Scalar;
    /// Calculates and returns the evolution of the yield stress.
    ///
    /// ```math
    /// \dot{Y} = \sqrt{\frac{2}{3}}\,H\,|\mathbf{D}_\mathrm{p}|
    /// ```
    fn yield_stress_evolution(
        &self,
        plastic_stretching_rate: &StretchingRatePlastic,
    ) -> Result<Scalar, ConstitutiveError> {
        Ok(self.hardening_slope() * plastic_stretching_rate.norm() * TWO_THIRDS.sqrt())
    }
}

/// The viscoplastic flow model.
#[derive(Clone, Debug)]
pub struct ViscoplasticFlow {
    /// The initial yield stress $`Y_0`$.
    pub initial_yield_stress: Scalar,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Scalar,
    /// The rate sensitivity parameter $`m`$.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate $`d_0`$.
    pub reference_flow_rate: Scalar,
}

impl Plastic for ViscoplasticFlow {
    fn initial_yield_stress(&self) -> Scalar {
        self.initial_yield_stress
    }
    fn hardening_slope(&self) -> Scalar {
        self.hardening_slope
    }
}

impl Viscoplastic for ViscoplasticFlow {
    fn rate_sensitivity(&self) -> Scalar {
        self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.reference_flow_rate
    }
}
