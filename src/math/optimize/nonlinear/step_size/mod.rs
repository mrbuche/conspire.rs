#[cfg(test)]
mod test;

mod barzilai_borwein;

pub use barzilai_borwein::{BarzilaiBorwein, Step as BarzilaiBorweinStep};

use crate::math::{Erase, Scalar, Tensor};
use std::fmt::{Debug, Display};

/// Ways of estimating a step size from the last two iterates.
///
/// What an estimator is given are the products of the change in solution
/// $`s`$ and the change in residual $`y`$ between them, and the number of
/// estimates it has made before, so that it holds nothing of its own.
pub trait StepEstimator: Clone + Debug + Display {
    /// The step size suggested by $`s^Ts`$, $`s^Ty`$ and $`y^Ty`$.
    fn estimate(&self, ss: Scalar, sy: Scalar, yy: Scalar, iteration: usize) -> Scalar;
}

/// What a step size estimate is made from: the last solution and residual.
///
/// The change in each since they were last shown is what an estimator is
/// given the products of. Where the change says nothing, the step size is
/// left as it was.
pub(crate) struct Secant<X, F> {
    iteration: usize,
    solution: X,
    residual: Option<F>,
}

impl<X, F, E> Secant<X, F>
where
    X: Erase<Erased = E> + Tensor,
    F: Erase<Erased = E> + Tensor,
    E: Tensor,
{
    pub(crate) fn new(solution: &X) -> Self {
        Self {
            iteration: 0,
            solution: solution.clone(),
            residual: None,
        }
    }
    pub(crate) fn solution(&self) -> &X {
        &self.solution
    }
    pub(crate) fn step_size(
        &mut self,
        estimator: &impl StepEstimator,
        solution: &X,
        residual: &F,
        step_size: Scalar,
    ) -> Scalar {
        self.solution -= solution;
        let change = self.residual.get_or_insert_with(|| {
            let mut zero = residual.clone();
            zero *= 0.0;
            zero
        });
        *change -= residual;
        let trial = estimator.estimate(
            self.solution
                .erase()
                .full_contraction(self.solution.erase()),
            change.erase().full_contraction(self.solution.erase()),
            change.erase().full_contraction(change.erase()),
            self.iteration,
        );
        *change = residual.clone();
        self.solution = solution.clone();
        self.iteration += 1;
        if trial.abs() > 0.0 && !trial.is_nan() {
            trial.abs()
        } else {
            step_size
        }
    }
}
