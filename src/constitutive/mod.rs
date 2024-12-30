//! Constitutive model library.

#[cfg(test)]
pub mod test;

pub mod cohesive;
pub mod fluid;
pub mod hybrid;
pub mod multiphysics;
pub mod solid;
pub mod thermal;

use crate::{
    get_defeat_message,
    math::optimize::OptimizeError,
    mechanics::{DeformationGradient, Scalar},
};
use std::fmt;

/// Array of constitutive model parameters.
pub type Parameters<'a> = &'a [Scalar];

/// Required methods for constitutive models.
pub trait Constitutive<'a> {
    /// Constructs and returns a new constitutive model.
    fn new(parameters: Parameters<'a>) -> Self;
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
        write!(
            f,
            "\n{}\n\x1b[0;2;31m{}\x1b[0m\n",
            error,
            get_defeat_message()
        )
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
        write!(f, "{}\x1b[0m", error)
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
