//! Plastic fluid constitutive models.

#[cfg(test)]
mod test;

pub mod hardening;
pub mod surface;

pub use hardening::{Linear, PlasticHardening, Voce};
pub use surface::{Hill, VonMises, YieldSurface};

use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, TensorArray, TensorTuple, TensorTupleVec},
    mechanics::{
        DeformationGradientPlastic, FlowDirectionPlastic, MandelStressElastic,
        StretchingRatePlastic,
    },
    units::{Dissipation, Rate, Stress},
};

/// Rate-independent plastic state variables $`(\mathbf{F}_\mathrm{p},\,\varepsilon_\mathrm{p})`$.
pub type PlasticStateVariables = TensorTuple<DeformationGradientPlastic, Quantity>;

/// The history of the rate-independent plastic state variables.
pub type PlasticStateVariablesHistory = TensorTupleVec<DeformationGradientPlastic, Quantity>;

/// Required methods for rate-independent (yield-surface) plastic fluid constitutive models.
///
/// Such a model is a [`YieldSurface`] combined with a [`PlasticHardening`] law, as
/// [`PlasticFlow`] does.
pub trait RateIndependentPlastic
where
    Self: PlasticHardening + YieldSurface,
{
    /// Returns the initial state of the variables.
    fn initial_state(&self) -> PlasticStateVariables {
        (DeformationGradientPlastic::identity(), Quantity::default()).into()
    }
    /// Calculates and returns the yield function.
    ///
    /// ```math
    /// f(\mathbf{M}_\mathrm{e}',\varepsilon_\mathrm{p}) = \phi(\mathbf{M}_\mathrm{e}') - Y(\varepsilon_\mathrm{p})
    /// ```
    fn yield_function(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(self.equivalent_stress(deviatoric_mandel_stress)?
            - self.yield_stress(equivalent_plastic_strain)?)
    }
    /// Calculates and returns the plastic stretching rate.
    ///
    /// ```math
    /// \mathbf{D}_\mathrm{p} = \dot{\gamma}\,\mathbf{N},\qquad
    /// \dot{\gamma}\geq 0,\quad f\leq 0,\quad \dot{\gamma}f = 0
    /// ```
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        plastic_multiplier: Quantity<Rate>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        Ok(self.flow_direction(deviatoric_mandel_stress)? * plastic_multiplier)
    }
}

/// The rate-independent plastic flow model: a yield surface $`S`$ combined with a
/// hardening law $`H`$.
#[derive(Clone, Debug)]
pub struct PlasticFlow<S, H> {
    /// The yield surface.
    pub surface: S,
    /// The hardening law.
    pub hardening: H,
}

impl<S, H> PlasticHardening for PlasticFlow<S, H>
where
    S: YieldSurface,
    H: PlasticHardening,
{
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.hardening.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening.hardening_slope()
    }
    fn yield_stress(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        self.hardening.yield_stress(equivalent_plastic_strain)
    }
    fn hardening_modulus(
        &self,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        self.hardening.hardening_modulus(equivalent_plastic_strain)
    }
}

impl<S, H> YieldSurface for PlasticFlow<S, H>
where
    S: YieldSurface,
    H: PlasticHardening,
{
    fn equivalent_stress(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        self.surface.equivalent_stress(deviatoric_mandel_stress)
    }
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        self.surface.flow_direction(deviatoric_mandel_stress)
    }
    fn flow_direction_slope(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        increment: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        self.surface
            .flow_direction_slope(deviatoric_mandel_stress, increment)
    }
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        self.surface
            .dissipation_potential(plastic_stretching_rate, yield_stress)
    }
}

impl<S, H> RateIndependentPlastic for PlasticFlow<S, H>
where
    S: YieldSurface,
    H: PlasticHardening,
{
}
