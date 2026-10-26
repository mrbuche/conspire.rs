#[cfg(test)]
mod test;

mod barzilai_borwein;

pub use barzilai_borwein::{BarzilaiBorwein, Step as BarzilaiBorweinStep};

use crate::math::{ContractWith, Quantity, Scalar, Tensor};
use std::{
    fmt::{Debug, Display},
    ops::Div,
};

/// Ways of estimating a step size from the last two iterates.
///
/// What an estimator is given are the products of the change in solution
/// $`s`$ and the change in residual $`y`$ between them, and the number of
/// estimates it has made before, so that it holds nothing of its own. The
/// products carry the units of what they contract, and the step comes out in
/// the unit that takes a residual to a solution.
pub trait StepEstimator: Clone + Debug + Display {
    /// The step size suggested by $`s^Ts`$, $`s^Ty`$ and $`y^Ty`$.
    fn estimate<SS, SY, YY, T>(&self, ss: SS, sy: SY, yy: YY, iteration: usize) -> T
    where
        SS: Div<SY, Output = T>,
        SY: Div<YY, Output = T>;
}

/// A solution whose changes, with those of the residual it is paired with,
/// make a step size.
///
/// This holds the units of the contractions, so that a solver names one bound
/// for them rather than the whole of the algebra.
pub trait SecantWith<F>: Tensor {
    /// The step size, in the unit that takes the residual to the solution.
    type Step;
    /// The step estimated from the change in solution and in residual.
    fn estimate_step(
        estimator: &impl StepEstimator,
        solution: &Self,
        residual: &F,
        iteration: usize,
    ) -> Self::Step;
}

impl<X, F, SS, SY, YY, U> SecantWith<F> for X
where
    X: ContractWith<X, Output = SS> + ContractWith<F, Output = SY> + Tensor,
    F: ContractWith<F, Output = YY>,
    SS: Div<SY, Output = Quantity<U>>,
    SY: Div<YY, Output = Quantity<U>>,
{
    type Step = Quantity<U>;
    fn estimate_step(
        estimator: &impl StepEstimator,
        solution: &X,
        residual: &F,
        iteration: usize,
    ) -> Self::Step {
        estimator.estimate(
            solution.contract_with(solution),
            solution.contract_with(residual),
            residual.contract_with(residual),
            iteration,
        )
    }
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

impl<X, F, U> Secant<X, F>
where
    X: SecantWith<F, Step = Quantity<U>>,
    F: Tensor,
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
        step_size: Quantity<U>,
    ) -> Quantity<U> {
        self.solution -= solution;
        let change = self.residual.get_or_insert_with(|| {
            let mut zero = residual.clone();
            zero *= 0.0;
            zero
        });
        *change -= residual;
        let trial = X::estimate_step(estimator, &self.solution, change, self.iteration);
        *change = residual.clone();
        self.solution = solution.clone();
        self.iteration += 1;
        let value: Scalar = trial.value();
        if value.abs() > 0.0 && !value.is_nan() {
            Quantity::new(value.abs())
        } else {
            step_size
        }
    }
}
