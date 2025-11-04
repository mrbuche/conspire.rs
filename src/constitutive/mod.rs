//! Constitutive model library.

#[cfg(test)]
pub mod test;

pub mod fluid;
pub mod hybrid;
pub mod multiphysics;
pub mod solid;
pub mod thermal;

use crate::{
    defeat_message,
    math::{Scalar, TensorError, TestError},
    mechanics::{Deformation, DeformationError, DeformationGradientGeneral},
};
use std::fmt::{self, Debug, Display, Formatter};

/// Required methods for constitutive models.
pub trait Constitutive
where
    Self: Debug,
{
    /// Calculates and returns the Jacobian.
    fn jacobian<const I: usize, const J: usize>(
        &self,
        deformation_gradient: &DeformationGradientGeneral<I, J>,
    ) -> Result<Scalar, ConstitutiveError> {
        match deformation_gradient.jacobian() {
            Err(DeformationError::InvalidJacobian(jacobian)) => Err(
                ConstitutiveError::InvalidJacobian(jacobian, format!("{self:?}")),
            ),
            Ok(jacobian) => Ok(jacobian),
        }
    }
}

/// Possible errors encountered in constitutive models.
pub enum ConstitutiveError {
    Custom(String, String),
    InvalidJacobian(Scalar, String),
    Upstream(String, String),
}

impl From<ConstitutiveError> for String {
    fn from(error: ConstitutiveError) -> Self {
        match error {
            ConstitutiveError::InvalidJacobian(jacobian, constitutive_model) => format!(
                "\x1b[1;91mInvalid Jacobian: {jacobian:.6e}.\x1b[0;91m\n\
                        In constitutive model: {constitutive_model}."
            ),
            _ => todo!(),
        }
    }
}

impl From<ConstitutiveError> for TestError {
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
            format!("unknown (temporary error handling)"),
        )
    }
}

impl Debug for ConstitutiveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Custom(message, constitutive_model) => format!(
                "\x1b[1;91m{message}\x1b[0;91m\n\
                 In constitutive model: {constitutive_model}."
            ),
            Self::InvalidJacobian(jacobian, constitutive_model) => {
                format!(
                    "\x1b[1;91mInvalid Jacobian: {jacobian:.6e}.\x1b[0;91m\n\
                    In constitutive model: {constitutive_model}."
                )
            }
            Self::Upstream(error, constitutive_model) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In constitutive model: {constitutive_model}."
                )
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl Display for ConstitutiveError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Custom(message, constitutive_model) => format!(
                "\x1b[1;91m{message}\x1b[0;91m\n\
                 In constitutive model: {constitutive_model}."
            ),
            Self::InvalidJacobian(jacobian, constitutive_model) => {
                format!(
                    "\x1b[1;91mInvalid Jacobian: {jacobian:.6e}.\x1b[0;91m\n\
                    In constitutive model: {constitutive_model}."
                )
            }
            Self::Upstream(error, constitutive_model) => {
                format!(
                    "{error}\x1b[0;91m\n\
                    In constitutive model: {constitutive_model}."
                )
            }
        };
        write!(f, "{error}\x1b[0m")
    }
}

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
