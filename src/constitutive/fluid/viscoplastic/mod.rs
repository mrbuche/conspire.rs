//! Viscoplastic fluid constitutive models.

use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::Plastic},
    math::{Rank2, Scalar, Tensor, TensorArray, TensorTuple, TensorTupleVec},
    mechanics::{DeformationGradientPlastic, MandelStressElastic, StretchingRatePlastic},
};

/// Viscoplastic state variables.
pub type ViscoplasticStateVariables<Y> = TensorTuple<DeformationGradientPlastic, Y>;

/// Viscoplastic state variables history.
pub type ViscoplasticStateVariablesHistory<Y> = TensorTupleVec<DeformationGradientPlastic, Y>;

/// Required methods for viscoplastic fluid constitutive models.
pub trait Viscoplastic<Y>
where
    Self: Plastic,
    Y: Tensor,
{
    /// Returns the initial state of the variables.
    fn initial_state(&self) -> ViscoplasticStateVariables<Y>;
    /// Calculates and returns the plastic evolution.
    ///
    /// ```math
    /// \dot{\mathbf{F}}_\mathrm{p} = \mathbf{D}_\mathrm{p}\cdot\mathbf{F}_\mathrm{p}\quad\text{and}\quad\dot{\varepsilon}_\mathrm{p} = |\mathbf{D}_\mathrm{p}|
    /// ```
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Y>,
    ) -> Result<ViscoplasticStateVariables<Y>, ConstitutiveError>;
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

impl Viscoplastic<Scalar> for ViscoplasticFlow {
    fn initial_state(&self) -> ViscoplasticStateVariables<Scalar> {
        (DeformationGradientPlastic::identity(), 0.0).into()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Scalar>,
    ) -> Result<ViscoplasticStateVariables<Scalar>, ConstitutiveError> {
        default_plastic_evolution(self, mandel_stress, state_variables)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> Scalar {
        self.reference_flow_rate
    }
}

pub fn default_plastic_evolution<C>(
    model: &C,
    mandel_stress: MandelStressElastic,
    state_variables: &ViscoplasticStateVariables<Scalar>,
) -> Result<ViscoplasticStateVariables<Scalar>, ConstitutiveError>
where
    C: Viscoplastic<Scalar>,
{
    let (deformation_gradient_p, &equivalent_plastic_strain) = state_variables.into();
    let plastic_stretching_rate = model.plastic_stretching_rate(
        mandel_stress.deviatoric(),
        model.yield_stress(equivalent_plastic_strain)?,
    )?;
    let equivalent_plastic_strain_rate = plastic_stretching_rate.norm();
    Ok((
        plastic_stretching_rate * deformation_gradient_p,
        equivalent_plastic_strain_rate,
    )
        .into())
}
