//! Plastic fluid constitutive models.

#[cfg(test)]
mod test;

use crate::{
    constitutive::ConstitutiveError,
    math::{Quantity, Tensor, TensorArray, TensorTuple},
    mechanics::{
        DeformationGradientPlastic, FlowDirectionPlastic, MandelStressElastic,
        StretchingRatePlastic,
    },
    units::{Dissipation, Rate, Stress},
};
use std::fmt::Debug;

/// Rate-independent plastic state variables $`(\mathbf{F}_\mathrm{p},\,\varepsilon_\mathrm{p})`$.
pub type PlasticStateVariables = TensorTuple<DeformationGradientPlastic, Quantity>;

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
        Ok(self.initial_yield_stress() + self.hardening_slope() * equivalent_plastic_strain)
    }
}

/// Required methods for rate-independent (yield-surface) plastic fluid constitutive models.
pub trait RateIndependentPlastic
where
    Self: Plastic,
{
    /// Returns the initial state of the variables.
    fn initial_state(&self) -> PlasticStateVariables {
        (DeformationGradientPlastic::identity(), Quantity::default()).into()
    }
    /// Calculates and returns the von Mises yield function.
    ///
    /// ```math
    /// f(\mathbf{M}_\mathrm{e}',\varepsilon_\mathrm{p}) = |\mathbf{M}_\mathrm{e}'| - Y(\varepsilon_\mathrm{p})
    /// ```
    fn yield_function(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
        equivalent_plastic_strain: Quantity,
    ) -> Result<Quantity<Stress>, ConstitutiveError> {
        Ok(deviatoric_mandel_stress.norm() - self.yield_stress(equivalent_plastic_strain)?)
    }
    /// Calculates and returns the associative plastic flow direction.
    ///
    /// ```math
    /// \mathbf{N} = \frac{\mathbf{M}_\mathrm{e}'}{|\mathbf{M}_\mathrm{e}'|}
    /// ```
    fn flow_direction(
        &self,
        deviatoric_mandel_stress: &MandelStressElastic,
    ) -> Result<FlowDirectionPlastic, ConstitutiveError> {
        let magnitude = deviatoric_mandel_stress.norm();
        if magnitude.is_zero() {
            Ok(FlowDirectionPlastic::zero())
        } else {
            Ok(deviatoric_mandel_stress / magnitude)
        }
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
    /// Calculates and returns the plastic dissipation potential.
    ///
    /// ```math
    /// \phi(\mathbf{D}_\mathrm{p}) = Y\,|\mathbf{D}_\mathrm{p}|
    /// ```
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(yield_stress * plastic_stretching_rate.norm())
    }
}

/// The rate-independent von Mises plastic flow model.
#[derive(Clone, Debug)]
pub struct PlasticFlow {
    /// The initial yield stress $`Y_0`$.
    pub yield_stress: Quantity<Stress>,
    /// The isotropic hardening slope $`H`$.
    pub hardening_slope: Quantity<Stress>,
}

impl Plastic for PlasticFlow {
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.yield_stress
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.hardening_slope
    }
}

impl RateIndependentPlastic for PlasticFlow {}
