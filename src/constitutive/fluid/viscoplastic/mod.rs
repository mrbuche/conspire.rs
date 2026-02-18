//! Viscoplastic fluid constitutive models.

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::plastic::{Plastic, StateVariables},
    },
    math::{Rank2, Scalar, Tensor, TensorArray},
    mechanics::{DeformationGradientPlastic, MandelStressElastic, StretchingRatePlastic},
};

/// Required methods for viscoplastic fluid constitutive models.
pub trait Viscoplastic
where
    Self: Plastic,
{
    /// Calculates and returns the plastic evolution.
    ///
    /// ```math
    /// \dot{\mathbf{F}}_\mathrm{p} = \mathbf{D}_\mathrm{p}\cdot\mathbf{F}_\mathrm{p}\quad\text{and}\quad\dot{\varepsilon}_\mathrm{p} = |\mathbf{D}_\mathrm{p}|
    /// ```
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        deformation_gradient_p: &DeformationGradientPlastic,
        equivalent_plastic_strain: Scalar,
    ) -> Result<StateVariables, ConstitutiveError> {
        let plastic_stretching_rate = self.plastic_stretching_rate(
            mandel_stress.deviatoric(),
            self.yield_stress(equivalent_plastic_strain)?,
        )?;
        let equivalent_plastic_strain_rate = plastic_stretching_rate.norm();
        Ok((
            plastic_stretching_rate * deformation_gradient_p,
            equivalent_plastic_strain_rate,
        )
            .into())
    }
    /// Calculates and returns the rate of plastic stretching.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = d_0\left(\frac{|\mathbf{M}_\mathrm{e}'|}{Y(S)}\right)^{\footnotesize\tfrac{1}{m}}\frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Scalar,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude == 0.0 {
            Ok(StretchingRatePlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress
                * (self.reference_flow_rate() / magnitude
                    * (magnitude / yield_stress).powf(1.0 / self.rate_sensitivity())))
        }
    }
    /// Returns the rate_sensitivity parameter.
    fn rate_sensitivity(&self) -> Scalar;
    /// Returns the reference flow rate.
    fn reference_flow_rate(&self) -> Scalar;
}

/// The viscoplastic flow model.
#[derive(Clone, Debug)]
pub struct ViscoplasticFlow {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Scalar,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Scalar,
    /// The rate sensitivity parameter $`m`$.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate $`d_0`$.
    pub reference_flow_rate: Scalar,
}

impl Plastic for ViscoplasticFlow {
    fn initial_yield_stress(&self) -> Scalar {
        self.yield_stress
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
