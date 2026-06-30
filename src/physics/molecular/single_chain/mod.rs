mod arbitrary;
mod efjc;
mod efrc;
mod fjc;
mod frc;
mod ideal;
mod swfjc;
mod ufjc;

/// Single-chain models of polymer statistical thermodynamics.
mod thermodynamics;

pub use arbitrary::{ArbitraryDiscrete, ArbitraryDiscretePotential};
pub use efjc::ExtensibleFreelyJointedChain;
pub use efrc::ExtensibleFreelyRotatingChain;
pub use fjc::FreelyJointedChain;
pub use frc::FreelyRotatingChain;
pub use ideal::IdealChain;
pub use swfjc::SquareWellFreelyJointedChain;
pub use thermodynamics::{
    Configuration, Ensemble, Isometric, Isotensional, IsotensionalExtensible, Legendre, MonteCarlo,
    MonteCarloExtensible, MonteCarloInextensible, Thermodynamics, ThermodynamicsExtensible,
};
pub use ufjc::ArbitraryPotentialFreelyJointedChain;

use crate::math::{Scalar, Style, StyledError, TestError, styled_error};
use std::fmt::Debug;

pub trait SingleChain
where
    Self: Clone + Debug,
{
    fn link_length(&self) -> Scalar;
    fn number_of_links(&self) -> u8;
}

pub trait Inextensible
where
    Self: SingleChain,
{
    fn maximum_nondimensional_extension(&self) -> Scalar;
    fn nondimensional_extension_check(
        &self,
        nondimensional_extension: Scalar,
    ) -> Result<(), SingleChainError> {
        if nondimensional_extension.abs() >= self.maximum_nondimensional_extension() {
            Err(SingleChainError::MaximumExtensibility(
                format!("{:?}", self.maximum_nondimensional_extension()),
                format!("{self:?}"),
            ))
        } else {
            Ok(())
        }
    }
}

pub trait Extensible
where
    Self: SingleChain,
{
}

pub enum SingleChainError {
    MaximumExtensibility(String, String),
    Upstream(String, String),
}

impl From<SingleChainError> for TestError {
    fn from(error: SingleChainError) -> Self {
        Self {
            message: error.to_string(),
        }
    }
}

impl From<SingleChainError> for String {
    fn from(error: SingleChainError) -> Self {
        error.message(&Style::detect())
    }
}

impl StyledError for SingleChainError {
    fn message(&self, style: &Style) -> String {
        let (h, c) = (style.headline, style.frame);
        match self {
            Self::MaximumExtensibility(maximum_nondimensional_extension, single_chain_model) => {
                format!(
                    "{h}Maximum extensibility ({maximum_nondimensional_extension}) reached.{c}\n\
                    In single-chain model: {single_chain_model}."
                )
            }
            Self::Upstream(error, single_chain_model) => format!(
                "{error}{c}\n\
                In single-chain model: {single_chain_model}."
            ),
        }
    }
}

styled_error!(SingleChainError);
