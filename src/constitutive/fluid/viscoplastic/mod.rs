//! Viscoplastic fluid constitutive models.

use crate::{
    constitutive::{ConstitutiveError, fluid::plastic::Plastic},
    math::{
        Derivative, Differentiate, Quantity, Rank2, Scalar, Tensor, TensorArray, TensorTuple,
        TensorTupleVec,
    },
    mechanics::{
        DeformationGradientPlastic, DeformationGradientRatePlastic, MandelStressElastic,
        StretchingRatePlastic,
    },
    units::{Rate, Stress},
};

/// Viscoplastic state variables.
pub type ViscoplasticStateVariables<Y> = TensorTuple<DeformationGradientPlastic, Y>;

/// Viscoplastic state variables history.
pub type ViscoplasticStateVariablesHistory<Y> = TensorTupleVec<DeformationGradientPlastic, Y>;

/// The evolution of the viscoplastic state variables.
pub type ViscoplasticEvolution<Y> = Derivative<ViscoplasticStateVariables<Y>>;

/// The history of the evolution of the viscoplastic state variables.
pub type ViscoplasticEvolutionHistory<Y> =
    TensorTupleVec<DeformationGradientRatePlastic, Derivative<Y>>;

/// Required methods for viscoplastic fluid constitutive models.
pub trait Viscoplastic<Y>
where
    Self: Plastic,
    Y: Differentiate + Tensor,
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
    ) -> Result<ViscoplasticEvolution<Y>, ConstitutiveError>;
    /// Calculates and returns the rate of plastic stretching.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = d_0\left(\frac{|\mathbf{M}_\mathrm{e}'|}{Y(S)}\right)^{\footnotesize\tfrac{1}{m}}\frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude.is_zero() {
            Ok(StretchingRatePlastic::zero())
        } else {
            let reference_flow_rate = self.reference_flow_rate();
            Ok(deviatoric_mandel_stress
                * (reference_flow_rate / magnitude
                    * (magnitude / yield_stress).powf(1.0 / self.rate_sensitivity())))
        }
    }
    /// Returns the rate_sensitivity parameter.
    fn rate_sensitivity(&self) -> Scalar;
    /// Returns the reference flow rate.
    fn reference_flow_rate(&self) -> Quantity<Rate>;
}

/// The viscoplastic flow model.
#[derive(Clone, Debug)]
pub struct ViscoplasticFlow {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Quantity<Stress>,
    /// The rate sensitivity parameter $`m`$.
    pub rate_sensitivity: Scalar,
    /// The reference flow rate $`d_0`$.
    pub reference_flow_rate: Quantity<Rate>,
}

impl Plastic for ViscoplasticFlow {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}

impl Viscoplastic<Quantity> for ViscoplasticFlow {
    fn initial_state(&self) -> ViscoplasticStateVariables<Quantity> {
        (DeformationGradientPlastic::identity(), Quantity::default()).into()
    }
    fn plastic_evolution(
        &self,
        mandel_stress: MandelStressElastic,
        state_variables: &ViscoplasticStateVariables<Quantity>,
    ) -> Result<ViscoplasticEvolution<Quantity>, ConstitutiveError> {
        default_plastic_evolution(self, mandel_stress, state_variables)
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.rate_sensitivity
    }
    fn reference_flow_rate(&self) -> Quantity<Rate> {
        self.reference_flow_rate
    }
}

pub fn default_plastic_evolution<C>(
    model: &C,
    mandel_stress: MandelStressElastic,
    state_variables: &ViscoplasticStateVariables<Quantity>,
) -> Result<ViscoplasticEvolution<Quantity>, ConstitutiveError>
where
    C: Viscoplastic<Quantity>,
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
