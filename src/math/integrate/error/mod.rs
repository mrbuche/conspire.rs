#[cfg(test)]
mod test;

use crate::math::{Scalar, Style, TestError, defeat_message};
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
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::InconsistentInitialConditions => {
                format!("{h}The initial condition z_0 is not consistent with g(t_0, y_0).")
            }
            Self::InitialTimeNotLessThanFinalTime => {
                format!("{h}The initial time must precede the final time.")
            }
            Self::Intermediate(message) => message.to_string(),
            Self::LengthTimeLessThanTwo => {
                format!("{h}The time must contain at least two entries.")
            }
            Self::MinimumStepSizeReached(dt_min, integrator) => {
                format!(
                    "{h}Minimum time step ({dt_min:?}) reached.{c}\n\
                    In integrator: {integrator}."
                )
            }
            Self::MinimumStepSizeUpstream(dt_min, error, integrator) => {
                format!(
                    "{error}{c}\n\
                    Causing error: {h}Minimum time step ({dt_min:?}) reached.{c}\n\
                    In integrator: {integrator}."
                )
            }
            Self::TimeStepNotSet(t0, tf, integrator) => {
                format!(
                    "{h}A positive time step must be set within [{t0:?}, {tf:?}].{c}\n\
                    In integrator: {integrator}."
                )
            }
            Self::Upstream(error, integrator) => {
                format!(
                    "{error}{c}\n\
                    In integrator: {integrator}."
                )
            }
        }
    }
}

impl Debug for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let style = Style::detect();
        write!(
            f,
            "\n{}\n{}{}{}\n",
            self.message(&style),
            style.footer,
            defeat_message(),
            style.reset
        )
    }
}

impl Display for IntegrationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let style = Style::detect();
        write!(f, "{}{}", self.message(&style), style.reset)
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
