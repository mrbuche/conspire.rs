#[cfg(feature = "doc")]
pub mod doc;
#[cfg(test)]
mod test;

mod dae;
mod error;
mod field;
mod ode;
mod tableau;

use crate::math::{Quantity, QuantityVector, Scalar, TensorVec};
use crate::units::Time;

/// The variable of integration values.
pub type Times<T = Time> = QuantityVector<T>;

/// The values of a span stepped at a fixed step, the last step being shorter if the step does
/// not divide the span.
///
/// A span that is a whole number of steps up to rounding is that number of steps, since
/// rounding it up would end the values with the final one twice, or with a step that is only
/// rounding. The rounding is that of the values, which is large when they are large next to
/// the step, and of the count of the steps.
pub(crate) fn fixed_step_times<T>(t_0: Quantity<T>, t_f: Quantity<T>, dt: Quantity<T>) -> Times<T> {
    let steps = (t_f - t_0).value() / dt.value();
    let whole = steps.round();
    let magnitude = t_0.value().abs().max(t_f.value().abs()) / dt.value().abs();
    let max_steps = if (steps - whole).abs() <= 4.0 * Scalar::EPSILON * (magnitude + whole) {
        whole
    } else {
        steps.ceil()
    } as usize;
    let mut times: Times<T> = (0..max_steps.max(1))
        .map(|step| t_0 + dt * (step as Scalar))
        .collect();
    times.push(t_f);
    times
}

pub use error::IntegrationError;
pub use field::{
    EvolvedIncrement, EvolvedState, Flat, HermiteSegment, Integrable, List, Product,
    StateEvolution, Unimodular, integrate_euler, integrate_rkmk, integrate_rkmk_adaptive,
    integrate_rkmk_dae_adaptive, integrate_rkmk_dae_adaptive_minimize,
    integrate_rkmk_dae_adaptive_root, integrate_rkmk_state, integrate_rkmk_state_adaptive,
    interpolate_hermite, rkmk_dae_step, rkmk_dae_step_minimize, rkmk_dae_step_root, rkmk_step,
};
pub use tableau::{ButcherTableau, EmbeddedTableau, StabilityRegion};

pub use ode::explicit::variable_step::{
    bogacki_shampine::Tableau as BogackiShampineTableau,
    dormand_prince::Tableau as DormandPrinceTableau, verner_8::Tableau as Verner8Tableau,
    verner_9::Tableau as Verner9Tableau,
};

pub use dae::{
    ExplicitDaeMinimize, ExplicitDaeRoot, ImplicitDaeMinimize, ImplicitDaeRoot,
    explicit::variable_step::{
        explicit::{
            ExplicitDaeVariableStepExplicit, ExplicitDaeVariableStepExplicitMinimize,
            ExplicitDaeVariableStepExplicitRoot, ExplicitDaeVariableStepFirstSameAsLast,
        },
        implicit::{
            ImplicitDaeVariableStepExplicit, ImplicitDaeVariableStepExplicitMinimize,
            ImplicitDaeVariableStepExplicitRoot,
        },
    },
};
pub use ode::{
    FixedStep, OdeIntegrator, VariableStep,
    dynamics::{
        explicit::{
            ExplicitDynamics, ExplicitDynamicsCoupled, stacked::Stacked,
            velocity_verlet::VelocityVerlet,
        },
        implicit::{ImplicitDynamics, ImplicitDynamicsCoupled, newmark::Newmark},
    },
    explicit::{
        Explicit, Spectrum,
        fixed_step::{
            FixedStepExplicit, bogacki_shampine::BogackiShampine as BogackiShampineFixedStep,
            dormand_prince::DormandPrince as DormandPrinceFixedStep, euler::Euler, heun::Heun,
            midpoint::Midpoint, ralston::Ralston, verner_8::Verner8 as Verner8FixedStep,
            verner_9::Verner9 as Verner9FixedStep,
        },
        variable_step::{
            FreeInterpolant,
            VariableStepExplicit,
            VariableStepExplicitFirstSameAsLast,
            bogacki_shampine::BogackiShampine,
            dormand_prince::DormandPrince,
            // heun_euler::HeunEuler,
            // midpoint_euler::MidpointEuler,
            // ralston_euler::RalstonEuler,
            verner_8::Verner8,
            verner_9::Verner9,
        },
    },
    implicit::{
        Implicit, backward_euler::BackwardEuler, midpoint::Midpoint as ImplicitMidpoint,
        rates::ImplicitRates, trapezoidal::Trapezoidal,
    },
};

/// Alias for [`Euler`].
pub type Ode1 = Euler;

/// Alias for [`BackwardEuler`].
pub type Ode1be = BackwardEuler;

// /// Alias for [`HeunEuler`].
// pub type Ode12 = HeunEuler;

/// Alias for [`Heun`].
pub type Ode2 = Heun;

/// Alias for [`BogackiShampine`].
pub type Ode23 = BogackiShampine;

/// Alias for [`BogackiShampineFixedStep`].
pub type Ode3 = BogackiShampineFixedStep;

/// Alias for [`DormandPrince`].
pub type Ode45 = DormandPrince;

/// Alias for [`DormandPrinceFixedStep`].
pub type Ode5 = DormandPrinceFixedStep;

/// Alias for [`Verner8`].
pub type Ode78 = Verner8;

/// Alias for [`Verner8FixedStep`].
pub type Ode8 = Verner8FixedStep;

/// Alias for [`Verner9`].
pub type Ode89 = Verner9;

/// Alias for [`Verner9FixedStep`].
pub type Ode9 = Verner9FixedStep;
