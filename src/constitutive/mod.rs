//! Constitutive model library.

#[cfg(test)]
pub mod test;

// pub mod fluid;
// pub mod hybrid;
// pub mod multiphysics;
pub mod solid;
// pub mod thermal;

use crate::{
    defeat_message,
    math::optimize::OptimizeError,
    mechanics::{Deformation, DeformationError, DeformationGradient, Scalar},
};
use std::{fmt, ops::Index};

/// Methods for lists of constitutive model parameters.
pub trait Parameters
where
    Self: fmt::Debug,
{
    fn get(&self, index: usize) -> &Scalar;
}

impl<const N: usize> Parameters for [Scalar; N] {
    fn get(&self, index: usize) -> &Scalar {
        self.index(index)
    }
}

impl<const N: usize> Parameters for &[Scalar; N] {
    fn get(&self, index: usize) -> &Scalar {
        self.index(index)
    }
}

/// Required methods for constitutive models.
pub trait Constitutive<P>
where
    Self: fmt::Debug,
{
    /// Calculates and returns the Jacobian.
    fn jacobian(
        &self,
        deformation_gradient: &DeformationGradient,
    ) -> Result<Scalar, ConstitutiveError> {
        match deformation_gradient.jacobian() {
            Err(DeformationError::InvalidJacobian(jacobian, deformation_gradient)) => {
                Err(ConstitutiveError::InvalidJacobian(
                    jacobian,
                    deformation_gradient,
                    format!("{:?}", self),
                ))
            }
            Ok(jacobian) => Ok(jacobian),
        }
    }
    /// Constructs and returns a new constitutive model.
    fn new(parameters: P) -> Self;
}

/// Possible errors encountered in constitutive models.
pub enum ConstitutiveError {
    Custom(String, DeformationGradient, String),
    InvalidJacobian(Scalar, DeformationGradient, String),
    MaximumStepsReached(usize, String),
    NotMinimum(String, String),
}

impl From<ConstitutiveError> for OptimizeError {
    fn from(_error: ConstitutiveError) -> OptimizeError {
        todo!()
    }
}

impl From<OptimizeError> for ConstitutiveError {
    fn from(_error: OptimizeError) -> ConstitutiveError {
        todo!()
    }
}

impl fmt::Debug for ConstitutiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Custom(message, deformation_gradient, constitutive_model) => format!(
                "\x1b[1;91m{}\x1b[0;91m\n\
                 From deformation gradient: {}.\n\
                 In constitutive model: {}.",
                message, deformation_gradient, constitutive_model
            ),
            Self::InvalidJacobian(jacobian, deformation_gradient, constitutive_model) => {
                format!(
                    "\x1b[1;91mInvalid Jacobian: {:.6e}.\x1b[0;91m\n\
                    From deformation gradient: {}.\n\
                    In constitutive model: {}.",
                    jacobian, deformation_gradient, constitutive_model
                )
            }
            Self::MaximumStepsReached(steps, constitutive_model) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({}) reached.\x1b[0;91m\n\
                     In constitutive model: {}.",
                    steps, constitutive_model
                )
            }
            Self::NotMinimum(deformation_gradient, constitutive_model) => {
                format!(
                    "\x1b[1;91mThe obtained solution is not a minimum.\x1b[0;91m\n\
                     {}\nIn constitutive model: {}.",
                    deformation_gradient, constitutive_model
                )
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl fmt::Display for ConstitutiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let error = match self {
            Self::Custom(message, deformation_gradient, constitutive_model) => format!(
                "\x1b[1;91m{}\x1b[0;91m\n\
                 From deformation gradient: {}.\n\
                 In constitutive model: {}.",
                message, deformation_gradient, constitutive_model
            ),
            Self::InvalidJacobian(jacobian, deformation_gradient, constitutive_model) => {
                format!(
                    "\x1b[1;91mInvalid Jacobian: {:.6e}.\x1b[0;91m\n\
                    From deformation gradient: {}.\n\
                    In constitutive model: {}.",
                    jacobian, deformation_gradient, constitutive_model
                )
            }
            Self::MaximumStepsReached(steps, constitutive_model) => {
                format!(
                    "\x1b[1;91mMaximum number of steps ({}) reached.\x1b[0;91m\n\
                     In constitutive model: {}.",
                    steps, constitutive_model
                )
            }
            Self::NotMinimum(deformation_gradient, constitutive_model) => {
                format!(
                    "\x1b[1;91mThe obtained solution is not a minimum.\x1b[0;91m\n\
                     {}\nIn constitutive model: {}.",
                    deformation_gradient, constitutive_model
                )
            }
        };
        write!(f, "\n{}\n\x1b[0;2;31m{}\x1b[0m\n", error, defeat_message())
    }
}

impl PartialEq for ConstitutiveError {
    fn eq(&self, other: &Self) -> bool {
        match self {
            Self::Custom(a, b, c) => match other {
                Self::Custom(d, e, f) => a == d && b == e && c == f,
                _ => false,
            },
            Self::InvalidJacobian(a, b, c) => match other {
                Self::InvalidJacobian(d, e, f) => a == d && b == e && c == f,
                _ => false,
            },
            Self::MaximumStepsReached(_, _) => todo!(),
            Self::NotMinimum(_, _) => todo!(),
        }
    }
}
