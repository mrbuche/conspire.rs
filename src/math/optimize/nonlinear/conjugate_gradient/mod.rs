#[cfg(test)]
mod test;

use crate::math::{
    ContractWith, Jacobian, Norm, Quantity, Scalar, Solution, Tensor,
    optimize::{
        BacktrackingLineSearch, EqualityConstraint, EqualityConstraints, NoSearch, ObjectiveMerit,
        ObjectiveOf, Optimization, OptimizationError, StepSize, Tolerances, UNSUPPORTED,
        nonlinear::step_size::{BarzilaiBorwein, Secant, SecantWith, StepEstimator},
    },
    sparse::SparseSolver,
};
use crate::units::{Dimensionless, UnitDiv, UnitMul, UnitRatio};
use std::{
    fmt::{self, Debug, Formatter},
    ops::Mul,
};

const INITIAL_STEP_SIZE: Scalar = 1e-2;

/// The method of nonlinear conjugate gradients.
pub struct ConjugateGradient<S = NoSearch, Q = BarzilaiBorwein> {
    /// Absolute error tolerances.
    pub abs_tol: Tolerances,
    /// How the last direction is carried through.
    pub conjugacy: Conjugacy,
    /// Norm type for error evaluation.
    pub error_norm: Norm,
    /// Line search algorithm.
    pub line_search: S,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Relative error tolerance.
    pub rel_tol: Option<Scalar>,
    /// Step size estimator.
    pub step_estimator: Q,
}

/// Formulas for the weight of the last direction in the next.
///
/// Each is zero-floored, so that a direction which would not descend restarts
/// from the gradient.
#[derive(Clone, Copy, Debug, Default)]
pub enum Conjugacy {
    /// The ratio of the squared gradients, which never restarts by itself.
    FletcherReeves,
    /// The change in gradient taken against the last direction.
    HestenesStiefel,
    /// The change in gradient taken against the last gradient.
    #[default]
    PolakRibiere,
}

impl<S, Q, J, X> BacktrackingLineSearch<J, X> for ConjugateGradient<S, Q>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
{
    type Search = S;
    fn line_search(&self) -> &S {
        &self.line_search
    }
}

impl<S, Q> Debug for ConjugateGradient<S, Q>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConjugateGradient {{ abs_tol: {:?}, conjugacy: {:?}, line_search: {}, max_steps: {:?}, rel_tol: {:?}, step_estimator: {} }}",
            self.abs_tol,
            self.conjugacy,
            self.line_search,
            self.max_steps,
            self.rel_tol,
            self.step_estimator
        )
    }
}

impl ConjugateGradient {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::with(NoSearch)
    }
}

impl<S> Default for ConjugateGradient<S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with(S::default())
    }
}

impl<S, Q> ConjugateGradient<S, Q> {
    /// The same solver, searching along each direction with another line search.
    pub fn with_line_search<T>(self, line_search: T) -> ConjugateGradient<T, Q> {
        ConjugateGradient {
            abs_tol: self.abs_tol,
            conjugacy: self.conjugacy,
            error_norm: self.error_norm,
            line_search,
            max_steps: self.max_steps,
            rel_tol: self.rel_tol,
            step_estimator: self.step_estimator,
        }
    }
    /// The same solver, estimating each step size with another estimator.
    pub fn with_step_estimator<T>(self, step_estimator: T) -> ConjugateGradient<S, T> {
        ConjugateGradient {
            abs_tol: self.abs_tol,
            conjugacy: self.conjugacy,
            error_norm: self.error_norm,
            line_search: self.line_search,
            max_steps: self.max_steps,
            rel_tol: self.rel_tol,
            step_estimator,
        }
    }
}

impl<S> ConjugateGradient<S> {
    fn with(line_search: S) -> Self {
        Self {
            abs_tol: Tolerances::default(),
            conjugacy: Conjugacy::default(),
            error_norm: Norm::Chebyshev,
            line_search,
            max_steps: 250,
            rel_tol: None,
            step_estimator: BarzilaiBorwein::default(),
        }
    }
}

impl<S, Q, F, J, X, A> Optimization<F, J, (), X> for ConjugateGradient<S, Q>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: Into<ObjectiveOf<J, X, A>>,
    J: ContractWith<J, Output = Quantity<A>> + Jacobian + Mul<StepSize<J, X>, Output = X>,
    J: Mul<Quantity<Dimensionless>, Output = J>,
    for<'a> &'a J: Mul<StepSize<J, X>, Output = X>,
    A: UnitDiv<A, Output = Dimensionless>,
    StepSize<J, X>: Mul<Quantity<Dimensionless>, Output = StepSize<J, X>>,
    X: Jacobian + Solution + SecantWith<J, Step = StepSize<J, X>>,
    <J as Tensor>::Unit: UnitRatio,
    <X as Tensor>::Unit: UnitDiv<<J as Tensor>::Unit>,
    <<X as Tensor>::Unit as UnitDiv<<J as Tensor>::Unit>>::Output: UnitMul<A>,
{
    fn minimize(
        &self,
        mut function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        _hessian: impl FnMut(&X) -> Result<(), String>,
        mut initial_guess: X,
        equality_constraint: EqualityConstraints,
        _sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        let objective = move |argument: &X| function(argument).map(Into::into);
        let Some(equality_constraint) =
            EqualityConstraint::lower_single(equality_constraint, &mut initial_guess)
        else {
            return Err(OptimizationError::Intermediate(UNSUPPORTED.to_string()));
        };
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => {
                descend(self, objective, jacobian, initial_guess, Some(&indices))
            }
            EqualityConstraint::Prescribed(_) => {
                unreachable!("Prescribed constraints are lowered first.")
            }
            EqualityConstraint::Multipliers(_) | EqualityConstraint::Linear(..) => {
                Err(OptimizationError::Intermediate(UNSUPPORTED.to_string()))
            }
            EqualityConstraint::None => descend(self, objective, jacobian, initial_guess, None),
        }
    }
}

fn restart_if_unbounded(weight: Quantity<Dimensionless>) -> Quantity<Dimensionless> {
    if weight > Quantity::new(0.0) && weight < Quantity::new(Scalar::INFINITY) {
        weight
    } else {
        Quantity::new(0.0)
    }
}

fn descend<S, Q, F, X, A>(
    conjugate_gradient: &ConjugateGradient<S, Q>,
    mut function: impl FnMut(&X) -> Result<ObjectiveOf<F, X, A>, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    indices: Option<&[usize]>,
) -> Result<X, OptimizationError>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: ContractWith<F, Output = Quantity<A>> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    F: Mul<Quantity<Dimensionless>, Output = F>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    A: UnitDiv<A, Output = Dimensionless>,
    StepSize<F, X>: Mul<Quantity<Dimensionless>, Output = StepSize<F, X>>,
    X: Jacobian + Solution + SecantWith<F, Step = StepSize<F, X>>,
    <F as Tensor>::Unit: UnitRatio,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    <<X as Tensor>::Unit as UnitDiv<<F as Tensor>::Unit>>::Output: UnitMul<A>,
{
    let mut gradient = |argument: &X| -> Result<F, String> {
        let mut residual = jacobian(argument)?;
        if let Some(indices) = indices {
            residual.zero_out(indices);
        }
        Ok(residual)
    };
    let mut previous: Option<(F, F)> = None;
    let mut relative_scale = None;
    let mut secant = Secant::new(&initial_guess);
    let mut solution = initial_guess;
    let mut step_size = StepSize::<F, X>::new(INITIAL_STEP_SIZE);
    let mut steps = 0;
    loop {
        let residual = gradient(&solution)?;
        let residual_norm = conjugate_gradient.error_norm.apply(&residual);
        let scale = *relative_scale.get_or_insert(residual_norm);
        if residual_norm < conjugate_gradient.abs_tol.residual() {
            return Ok(solution);
        } else if let Some(rel_tol) = conjugate_gradient.rel_tol
            && Scalar::from(residual_norm / scale) < rel_tol
        {
            return Ok(solution);
        } else if steps == conjugate_gradient.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                conjugate_gradient.max_steps,
                format!("{conjugate_gradient:?}"),
            ));
        }
        steps += 1;
        let squared = residual.contract_with(&residual);
        let mut direction = match previous.take() {
            Some((last_residual, last_direction)) => {
                let last_squared = last_residual.contract_with(&last_residual);
                let change = squared - residual.contract_with(&last_residual);
                let weight = match conjugate_gradient.conjugacy {
                    Conjugacy::FletcherReeves => squared / last_squared,
                    Conjugacy::HestenesStiefel => {
                        change
                            / (last_direction.contract_with(&residual)
                                - last_direction.contract_with(&last_residual))
                    }
                    Conjugacy::PolakRibiere => change / last_squared,
                };
                let weight = restart_if_unbounded(weight);
                last_direction * weight + &residual
            }
            None => residual.clone(),
        };
        if residual.contract_with(&direction) <= Quantity::new(0.0) {
            direction = residual.clone();
        }
        step_size = secant.step_size(
            &conjugate_gradient.step_estimator,
            &solution,
            &residual,
            step_size,
        );
        let scale = (squared / direction.contract_with(&direction)).sqrt();
        let taken = conjugate_gradient.backtracking_line_search(
            &mut function,
            &mut gradient,
            &solution,
            &residual,
            &direction,
            step_size * scale,
        )?;
        solution -= &direction * taken;
        previous = Some((residual, direction));
    }
}
