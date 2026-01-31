#[cfg(feature = "doc")]
pub mod doc;

#[cfg(test)]
mod test;

mod ode;

pub use ode::{
    FixedStep, OdeSolver, VariableStep,
    explicit::{
        Explicit,
        fixed_step::{
            FixedStepExplicit, bogacki_shampine::BogackiShampine as BogackiShampineFixedStep,
            dormand_prince::DormandPrince as DormandPrinceFixedStep, euler::Euler, heun::Heun,
            midpoint::Midpoint, ralston::Ralston,
        },
        internal_variables::ExplicitInternalVariables,
        variable_step::{
            VariableStepExplicit,
            VariableStepExplicitFirstSameAsLast,
            bogacki_shampine::BogackiShampine,
            dormand_prince::DormandPrince,
            internal_variables::{
                VariableStepExplicitInternalVariables,
                VariableStepExplicitInternalVariablesFirstSameAsLast,
            },
            // heun_euler::HeunEuler,
            // midpoint_euler::MidpointEuler,
            // ralston_euler::RalstonEuler,
            verner_8::Verner8,
            verner_9::Verner9,
        },
    },
    implicit::{
        ImplicitFirstOrder, ImplicitZerothOrder, backward_euler::BackwardEuler,
        midpoint::Midpoint as ImplicitMidpoint, trapezoidal::Trapezoidal,
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

/// Alias for [`Verner9`].
pub type Ode89 = Verner9;

use crate::{
    defeat_message,
    math::{Scalar, TestError},
};
use std::fmt::{self, Debug, Display, Formatter};

/// Possible errors encountered when integrating.
pub enum IntegrationError {
    InconsistentInitialConditions,
    InitialTimeNotLessThanFinalTime,
    Intermediate(String),
    LengthTimeLessThanTwo,
    MinimumStepSizeReached(Scalar, String),
    MinimumStepSizeUpstream(Scalar, String, String),
    TimeStepNotSet(Scalar, Scalar, String),
    Upstream(String, String),
}

impl From<String> for IntegrationError {
    fn from(error: String) -> Self {
        Self::Intermediate(error)
    }
}

impl Debug for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InconsistentInitialConditions => {
                "\x1b[1;91mThe initial condition z_0 is not consistent with g(t_0, y_0)."
                    .to_string()
            }
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
            }
            Self::MinimumStepSizeReached(dt_min, integrator) => {
                format!(
                    "\x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::MinimumStepSizeUpstream(dt_min, error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    Causing error: \x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::TimeStepNotSet(t0, tf, integrator) => {
                format!(
                    "\x1b[1;91mA positive time step must be set within [{t0:?}, {tf:?}].\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::Upstream(error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::InconsistentInitialConditions => {
                "\x1b[1;91mThe initial condition z_0 is not consistent with g(t_0, y_0)."
                    .to_string()
            }
            Self::InitialTimeNotLessThanFinalTime => {
                "\x1b[1;91mThe initial time must precede the final time.".to_string()
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                "\x1b[1;91mThe time must contain at least two entries.".to_string()
            }
            Self::MinimumStepSizeReached(dt_min, integrator) => {
                format!(
                    "\x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::MinimumStepSizeUpstream(dt_min, error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    Causing error: \x1b[1;91mMinimum time step ({dt_min:?}) reached.\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::TimeStepNotSet(t0, tf, integrator) => {
                format!(
                    "\x1b[1;91mA positive time step must be set within [{t0:?}, {tf:?}].\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
            Self::Upstream(error, integrator) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In integrator: {integrator}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

impl From<IntegrationError> for String {
    fn from(error: IntegrationError) -> Self {
        format!("{}", error)
    }
}

impl From<IntegrationError> for TestError {
    fn from(error: IntegrationError) -> Self {
        TestError {
            message: error.to_string(),
        }
    }
}
