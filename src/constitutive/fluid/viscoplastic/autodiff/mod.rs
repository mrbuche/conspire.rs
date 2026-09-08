//! Autodiff-backed viscoplastic constitutive models (`std::autodiff` / Enzyme).
//!
//! A model that supplies its dual dissipation potential
//! $`\phi^*(\mathbf{M}_\mathrm{e}',Y)`$ as an `#[autodiff]`-differentiable scalar
//! kernel ([`AutodiffViscoplastic`]) gets the full `Viscoplastic` API by
//! wrapping it in [`Autodiff`]: the plastic stretching rate is reverse mode over
//! $`\phi^*`$ w.r.t. the deviatoric Mandel stress. Composing
//! `Canonical<E, Autodiff<V>>` then yields an autodiff elastic-viscoplastic
//! solid. Maintained models keep their hand-written flow rule.

#![allow(clippy::needless_range_loop)]

pub mod viscoplastic_flow;

pub use crate::constitutive::autodiff::Autodiff;
pub use viscoplastic_flow::AutodiffViscoplasticFlow;

use crate::{
    constitutive::{
        ConstitutiveError,
        fluid::{
            plastic::Plastic,
            viscoplastic::{
                Viscoplastic, ViscoplasticEvolution, ViscoplasticStateVariables,
                default_plastic_evolution,
            },
        },
    },
    math::{Quantity, Scalar, Tensor, TensorArray},
    mechanics::{DeformationGradientPlastic, MandelStressElastic, StretchingRatePlastic},
    units::{Dissipation, Rate, Stress},
};
use std::fmt::Debug;

/// A viscoplastic model given as `#[autodiff]`-differentiable scalar potential
/// kernels: the plastic stretching rate is reverse mode over the dual
/// dissipation potential w.r.t. the deviatoric Mandel stress.
pub trait AutodiffViscoplastic {
    fn parameters(&self) -> [f64; 2];
    fn initial_yield_stress(&self) -> Quantity<Stress>;
    fn hardening_slope(&self) -> Quantity<Stress>;
    fn rate_sensitivity(&self) -> Scalar;
    fn reference_flow_rate(&self) -> Quantity<Rate>;
    fn dissipation(
        parameters: &[f64; 2],
        plastic_stretching_rate: &[f64; 9],
        yield_stress: f64,
    ) -> f64;
    fn dual_dissipation(
        parameters: &[f64; 2],
        mandel_deviatoric: &[f64; 9],
        yield_stress: f64,
    ) -> f64;
    fn stretching_rate(
        parameters: &[f64; 2],
        mandel_deviatoric: &[f64; 9],
        yield_stress: f64,
        out: &mut [f64; 9],
    );
}

impl<M> Plastic for Autodiff<M>
where
    M: AutodiffViscoplastic + Clone + Debug,
{
    fn initial_yield_stress(&self) -> Quantity<Stress> {
        self.0.initial_yield_stress()
    }
    fn hardening_slope(&self) -> Quantity<Stress> {
        self.0.hardening_slope()
    }
}

impl<M> Viscoplastic<Quantity> for Autodiff<M>
where
    M: AutodiffViscoplastic + Clone + Debug,
{
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
    fn plastic_stretching_rate(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<StretchingRatePlastic, ConstitutiveError> {
        if deviatoric_mandel_stress.norm().is_zero() {
            Ok(StretchingRatePlastic::zero())
        } else {
            let mut out = [0.0; 9];
            M::stretching_rate(
                &self.0.parameters(),
                &deviatoric_mandel_stress.flatten(),
                yield_stress.value(),
                &mut out,
            );
            Ok(StretchingRatePlastic::unflatten(out))
        }
    }
    fn dissipation_potential(
        &self,
        plastic_stretching_rate: StretchingRatePlastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(Quantity::new(M::dissipation(
            &self.0.parameters(),
            &plastic_stretching_rate.flatten(),
            yield_stress.value(),
        )))
    }
    fn dual_dissipation_potential(
        &self,
        deviatoric_mandel_stress: MandelStressElastic,
        yield_stress: Quantity<Stress>,
    ) -> Result<Quantity<Dissipation>, ConstitutiveError> {
        Ok(Quantity::new(M::dual_dissipation(
            &self.0.parameters(),
            &deviatoric_mandel_stress.flatten(),
            yield_stress.value(),
        )))
    }
    fn rate_sensitivity(&self) -> Scalar {
        self.0.rate_sensitivity()
    }
    fn reference_flow_rate(&self) -> Quantity<Rate> {
        self.0.reference_flow_rate()
    }
}
