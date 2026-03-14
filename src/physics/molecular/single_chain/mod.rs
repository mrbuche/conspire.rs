mod fjc;
mod thermodynamics;

pub use fjc::FreelyJointedChain;
pub use thermodynamics::{Ensemble, Isometric, Isotensional, Legendre, Thermodynamics};

use crate::math::Scalar;

pub trait SingleChain {
    fn number_of_links(&self) -> Scalar;
}
