use super::{LineSearchError, ResidualMerit, exhausted};
use crate::math::Scalar;
use std::fmt::{self, Display, Formatter};

/// The Armijo condition on half the squared norm of the residual.
#[derive(Clone, Debug)]
pub struct Residual {
    pub control: Scalar,
    pub cut_back: Scalar,
    pub max_steps: usize,
}

impl Default for Residual {
    fn default() -> Self {
        Self {
            control: 1e-3,
            cut_back: 9e-1,
            max_steps: 100,
        }
    }
}

impl Display for Residual {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Residual")
    }
}

impl ResidualMerit for Residual {
    fn search_residual(
        &self,
        mut merit: impl FnMut(Scalar) -> Result<Scalar, String>,
        value: Scalar,
        fraction: Scalar,
    ) -> Result<Scalar, LineSearchError> {
        let t = self.control * 2.0 * fraction * value;
        let mut n = 1.0;
        for _ in 0..self.max_steps {
            if let Ok(trial) = merit(n)
                && value - trial >= n * t
            {
                return Ok(n);
            } else {
                n *= self.cut_back
            }
        }
        Err(exhausted(self, self.max_steps))
    }
}
