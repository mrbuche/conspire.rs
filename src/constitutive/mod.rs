//! Constitutive model library.

#[cfg(test)]
pub mod test;

pub mod cohesive;
pub mod fluid;
pub mod hybrid;
pub mod multiphysics;
pub mod solid;
pub mod thermal;

use crate::math::{Scalar, Style, StyledError, TensorError, assert::AssertionError, styled_error};
use std::fmt::Debug;

/// Required methods for constitutive models.
pub trait Constitutive
where
    Self: Clone + Debug,
{
}

/// Possible errors encountered in constitutive models.
pub enum ConstitutiveError {
    Custom(String, String),
    InvalidJacobian(Scalar, String),
    Upstream(String, String),
}

impl From<ConstitutiveError> for AssertionError {
    fn from(error: ConstitutiveError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<TensorError> for ConstitutiveError {
    fn from(error: TensorError) -> Self {
        ConstitutiveError::Custom(
            error.to_string(),
            "unknown (temporary error handling)".to_string(),
        )
    }
}

impl From<ConstitutiveError> for String {
    fn from(error: ConstitutiveError) -> Self {
        error.message(&Style::detect())
    }
}

impl StyledError for ConstitutiveError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::Custom(message, constitutive_model) => format!(
                "{h}{message}{c}\n\
                In constitutive model: {constitutive_model}."
            ),
            Self::InvalidJacobian(jacobian, constitutive_model) => format!(
                "{h}Invalid Jacobian: {jacobian:.6e}.{c}\n\
                In constitutive model: {constitutive_model}."
            ),
            Self::Upstream(error, constitutive_model) => format!(
                "{error}{c}\n\
                In constitutive model: {constitutive_model}."
            ),
        }
    }
}

styled_error!(ConstitutiveError);

impl PartialEq for ConstitutiveError {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Custom(a, b) => match other {
                Self::Custom(c, d) => a == c && b == d,
                _ => false,
            },
            Self::InvalidJacobian(a, b) => match other {
                Self::InvalidJacobian(c, d) => a == c && b == d,
                _ => false,
            },
            Self::Upstream(a, b) => match other {
                Self::Upstream(c, d) => a == c && b == d,
                _ => false,
            },
        }
    }
}
