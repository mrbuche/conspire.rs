#[cfg(test)]
mod test;

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

impl IntegrationError {
    fn message(&self) -> String {
        match self {
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
        }
    }
}

impl Debug for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "\n{}\n\x1b[0;2;31m{}\x1b[0m\n",
            self.message(),
            defeat_message()
        )
    }
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}\x1b[0m", self.message())
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
