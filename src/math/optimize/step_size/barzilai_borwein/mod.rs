use super::StepEstimator;
use crate::math::Scalar;
use std::fmt::{self, Display, Formatter};

/// The step sizes of Barzilai and Borwein.
///
/// The step along which the change in residual best matches the change in
/// solution, which stands in for a line search where there is none.
#[derive(Clone, Debug, Default)]
pub struct BarzilaiBorwein {
    pub step: Step,
}

/// Which of the two step sizes is taken.
#[derive(Clone, Copy, Debug, Default)]
pub enum Step {
    /// The short step $`s^Ty/y^Ty`$.
    #[default]
    Short,
    /// The long step $`s^Ts/s^Ty`$.
    Long,
    /// The long step and the short step in turn.
    Alternating,
}

impl Display for BarzilaiBorwein {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "BarzilaiBorwein")
    }
}

impl StepEstimator for BarzilaiBorwein {
    fn estimate(&self, ss: Scalar, sy: Scalar, yy: Scalar, iteration: usize) -> Scalar {
        match self.step {
            Step::Short => sy / yy,
            Step::Long => ss / sy,
            Step::Alternating => {
                if iteration.is_multiple_of(2) {
                    ss / sy
                } else {
                    sy / yy
                }
            }
        }
    }
}
