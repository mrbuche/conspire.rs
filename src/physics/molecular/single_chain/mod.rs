mod fjc;
// mod ideal;
mod thermodynamics;

pub use fjc::FreelyJointedChain;
pub use thermodynamics::{Ensemble, Isometric, Isotensional, Legendre, Thermodynamics};

use crate::math::{Scalar, TestError};
use std::fmt::{self, Debug, Display, Formatter};

pub trait SingleChain
where
    Self: Clone + Debug,
{
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
            Err(SingleChainError::MaximumExtensibility)
        } else {
            Ok(())
        }
    }
}

#[derive(Debug)]
pub enum SingleChainError {
    MaximumExtensibility,
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
        Self::from(&error)
    }
}

impl From<&SingleChainError> for String {
    fn from(error: &SingleChainError) -> Self {
        match error {
            SingleChainError::MaximumExtensibility => {
                "\x1b[1;91mMaximum extensibility reached.\x1b[0;91m".to_string()
            }
            SingleChainError::Upstream(error, single_chain_model) => format!(
                "{error}\x1b[0;91m\n\
                    In single-chain model: {single_chain_model}."
            ),
        }
    }
}

impl Display for SingleChainError {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}\x1b[0m", String::from(self))
    }
}
