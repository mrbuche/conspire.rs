//! Constitutive models.

#[cfg(test)]
pub mod test;

#[cfg(feature = "autodiff")]
pub mod autodiff;

pub mod canonical;
pub mod chemical;
pub mod cohesive;
pub mod diffusion;
pub mod electrical;
pub mod fluid;
pub mod hybrid;
pub mod multiphysics;
pub mod solid;
pub mod thermal;

use crate::math::{Scalar, Style, StyledError, assert::AssertionError, styled_error};
use core::fmt::{Debug, Display};

/// Required methods for constitutive models.
pub trait Constitutive
where
    Self: Clone + Debug,
{
}

/// Possible errors encountered in constitutive models.
#[derive(PartialEq)]
pub enum ConstitutiveError {
    Custom(String, String),
    InvalidJacobian(Scalar, String),
    Upstream(String, String),
}

/// Without the standard library, which is how the library is built as code for a device, there
/// is no heap to make the text of an error in, so an error is its kind and its number, and
/// whatever runs the code on the host can say more.
#[cfg(not(feature = "std"))]
impl ConstitutiveError {
    pub fn custom(_: impl Display, _: &(impl Debug + ?Sized)) -> Self {
        Self::Custom(String::new(), String::new())
    }
    pub fn invalid_jacobian(jacobian: Scalar, _: &(impl Debug + ?Sized)) -> Self {
        Self::InvalidJacobian(jacobian, String::new())
    }
    pub fn upstream(_: impl Display, _: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(String::new(), String::new())
    }
}

#[cfg(feature = "std")]
impl ConstitutiveError {
    pub fn custom(message: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Custom(format!("{message}"), format!("{context:?}"))
    }
    pub fn invalid_jacobian(jacobian: Scalar, context: &(impl Debug + ?Sized)) -> Self {
        Self::InvalidJacobian(jacobian, format!("{context:?}"))
    }
    pub fn upstream(error: impl Display, context: &(impl Debug + ?Sized)) -> Self {
        Self::Upstream(format!("{error}"), format!("{context:?}"))
    }
}

impl From<ConstitutiveError> for AssertionError {
    fn from(error: ConstitutiveError) -> Self {
        Self {
            message: error.to_string(),
        }
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
