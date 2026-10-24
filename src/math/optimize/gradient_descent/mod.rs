#[cfg(test)]
mod test;

use super::{
    super::{Jacobian, Matrix, Scalar, Solution, Tensor, Vector, sparse::SparseSolver},
    BacktrackingLineSearch, EqualityConstraint, EqualityConstraints, NoSearch, ObjectiveMerit,
    Optimization, OptimizationError, RootFinding, StepSize, Tolerances, UNSUPPORTED, no_objective,
    step_size::{BarzilaiBorwein, Secant, StepEstimator},
};
use crate::math::{Erase, Is, Norm};
use crate::units::{UnitDiv, UnitMul, UnitSum};
use std::{
    fmt::{self, Debug, Formatter},
    ops::Mul,
};

const CUTBACK_FACTOR: Scalar = 0.8;
const CUTBACK_FACTOR_MINUS_ONE: Scalar = 1.0 - CUTBACK_FACTOR;
const INITIAL_STEP_SIZE: Scalar = 1e-2;

/// The method of gradient descent.
pub struct GradientDescent<S = NoSearch, Q = BarzilaiBorwein> {
    /// Absolute error tolerances.
    pub abs_tol: Tolerances,
    /// Lagrangian dual.
    pub dual: bool,
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

impl<S, Q, J, X> BacktrackingLineSearch<J, X> for GradientDescent<S, Q>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
{
    type Search = S;
    fn line_search(&self) -> &S {
        &self.line_search
    }
}

impl<S, Q> Debug for GradientDescent<S, Q>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GradientDescent {{ abs_tol: {:?}, dual: {:?}, line_search: {}, max_steps: {:?}, rel_tol: {:?}, step_estimator: {} }}",
            self.abs_tol,
            self.dual,
            self.line_search,
            self.max_steps,
            self.rel_tol,
            self.step_estimator
        )
    }
}

impl GradientDescent {
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::with(NoSearch)
    }
}

impl<S> Default for GradientDescent<S>
where
    S: Default,
{
    fn default() -> Self {
        Self::with(S::default())
    }
}

impl<S, Q> GradientDescent<S, Q> {
    /// The same solver, searching along each step with another line search.
    pub fn with_line_search<T>(self, line_search: T) -> GradientDescent<T, Q> {
        GradientDescent {
            abs_tol: self.abs_tol,
            dual: self.dual,
            error_norm: self.error_norm,
            line_search,
            max_steps: self.max_steps,
            rel_tol: self.rel_tol,
            step_estimator: self.step_estimator,
        }
    }
    /// The same solver, estimating each step size with another estimator.
    pub fn with_step_estimator<T>(self, step_estimator: T) -> GradientDescent<S, T> {
        GradientDescent {
            abs_tol: self.abs_tol,
            dual: self.dual,
            error_norm: self.error_norm,
            line_search: self.line_search,
            max_steps: self.max_steps,
            rel_tol: self.rel_tol,
            step_estimator,
        }
    }
}

impl<S> GradientDescent<S> {
    fn with(line_search: S) -> Self {
        Self {
            abs_tol: Tolerances::default(),
            dual: false,
            error_norm: Norm::Chebyshev,
            line_search,
            max_steps: 250,
            rel_tol: None,
            step_estimator: BarzilaiBorwein::default(),
        }
    }
}

impl<Q, F, X, E> RootFinding<F, (), X> for GradientDescent<NoSearch, Q>
where
    Q: StepEstimator,
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    fn root(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        _jacobian: impl FnMut(&X) -> Result<(), String>,
        mut initial_guess: X,
        equality_constraint: EqualityConstraints,
        _sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        let Some(equality_constraint) =
            EqualityConstraint::lower_single(equality_constraint, &mut initial_guess)
        else {
            return Err(OptimizationError::Intermediate(UNSUPPORTED.to_string()));
        };
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => {
                constrained_fixed(self, no_objective, function, initial_guess, indices)
            }
            EqualityConstraint::Prescribed(_) => {
                unreachable!("Prescribed constraints are lowered first.")
            }
            EqualityConstraint::Multipliers(_) => {
                Err(OptimizationError::Intermediate(UNSUPPORTED.to_string()))
            }
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                if self.dual {
                    constrained_dual(
                        self,
                        function,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                } else {
                    constrained(
                        self,
                        function,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                }
            }
            EqualityConstraint::None => {
                unconstrained(self, no_objective, function, initial_guess, None)
            }
        }
    }
}

impl<S, Q, F, J, X, E> Optimization<F, J, (), X> for GradientDescent<S, Q>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: Erase<Erased = Scalar> + Tensor,
    <J as Tensor>::Unit: UnitMul<<X as Tensor>::Unit>,
    <<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output: UnitSum,
    <<<J as Tensor>::Unit as UnitMul<<X as Tensor>::Unit>>::Output as UnitSum>::Output:
        Is<<F as Tensor>::Unit>,
    J: Erase<Erased = E> + Jacobian + Mul<StepSize<J, X>, Output = X>,
    for<'a> &'a J: Mul<StepSize<J, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<J as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
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
        let objective = move |argument: &X| function(argument).map(|value| *value.erase());
        let Some(equality_constraint) =
            EqualityConstraint::lower_single(equality_constraint, &mut initial_guess)
        else {
            return Err(OptimizationError::Intermediate(UNSUPPORTED.to_string()));
        };
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => {
                constrained_fixed(self, objective, jacobian, initial_guess, indices)
            }
            EqualityConstraint::Prescribed(_) => {
                unreachable!("Prescribed constraints are lowered first.")
            }
            EqualityConstraint::Multipliers(_) => {
                Err(OptimizationError::Intermediate(UNSUPPORTED.to_string()))
            }
            EqualityConstraint::Linear(constraint_matrix, constraint_rhs) => {
                if self.dual {
                    constrained_dual(
                        self,
                        jacobian,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                } else {
                    constrained(
                        self,
                        jacobian,
                        initial_guess,
                        constraint_matrix,
                        constraint_rhs,
                    )
                }
            }
            EqualityConstraint::None => {
                unconstrained(self, objective, jacobian, initial_guess, None)
            }
        }
    }
}

fn unconstrained<S, Q, F, X, E>(
    gradient_descent: &GradientDescent<S, Q>,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    linear_equality_constraint: Option<(&Matrix, &Vector)>,
) -> Result<X, OptimizationError>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
{
    let constraint = if let Some((constraint_matrix, multipliers)) = linear_equality_constraint {
        Some(multipliers * constraint_matrix)
    } else {
        None
    };
    let mut residual;
    let mut secant = Secant::new(&initial_guess);
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    let mut steps = 0;
    loop {
        residual = if let Some(ref extra) = constraint {
            jacobian(&solution)? - extra
        } else {
            jacobian(&solution)?
        };
        if gradient_descent.error_norm.apply(&residual) < gradient_descent.abs_tol.residual() {
            return Ok(solution);
        } else if steps == gradient_descent.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                gradient_descent.max_steps,
                format!("{gradient_descent:?}"),
            ));
        } else {
            steps += 1;
            step_size = secant.step_size(
                &gradient_descent.step_estimator,
                &solution,
                &residual,
                step_size,
            );
            step_size = gradient_descent.backtracking_line_search::<F, E>(
                |trial: &X, _: Scalar| function(trial),
                &mut jacobian,
                &solution,
                &residual,
                &residual,
                step_size,
            )?;
            solution -= residual * StepSize::<F, X>::new(step_size);
        }
    }
}

fn constrained_fixed<S, Q, F, X, E>(
    gradient_descent: &GradientDescent<S, Q>,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
{
    let mut relative_scale = 0.0;
    let mut residual: F;
    let mut residual_norm;
    let mut secant = Secant::new(&initial_guess);
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?;
        residual.zero_out(&indices);
        residual_norm = gradient_descent.error_norm.measure(&residual);
        if gradient_descent.rel_tol.is_some() && steps == 0 {
            relative_scale = gradient_descent.error_norm.measure(&residual)
        }
        if residual_norm < gradient_descent.abs_tol.residual {
            return Ok(solution);
        } else if let Some(rel_tol) = gradient_descent.rel_tol
            && residual_norm / relative_scale < rel_tol
        {
            return Ok(solution);
        } else if steps == gradient_descent.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                gradient_descent.max_steps,
                format!("{gradient_descent:?}"),
            ));
        } else {
            steps += 1;
            step_size = secant.step_size(
                &gradient_descent.step_estimator,
                &solution,
                &residual,
                step_size,
            );
            step_size = gradient_descent.backtracking_line_search::<F, E>(
                |trial: &X, _: Scalar| function(trial),
                &mut jacobian,
                &solution,
                &residual,
                &residual,
                step_size,
            )?;
            solution -= residual * StepSize::<F, X>::new(step_size);
        }
    }
}

fn constrained<S, Q, F, X, E>(
    gradient_descent: &GradientDescent<S, Q>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if S::SEARCHES {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let mut residual_solution;
    let mut secant_solution = Secant::new(&initial_guess);
    let mut solution = initial_guess;
    let mut step_size_solution = INITIAL_STEP_SIZE;
    let num_constraints = constraint_rhs.len();
    let mut residual_multipliers;
    let mut multipliers = Vector::zero(num_constraints);
    let mut secant_multipliers = Secant::new(&multipliers);
    let mut step_size_multipliers = INITIAL_STEP_SIZE;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual_solution = jacobian(&solution)? - &multipliers * &constraint_matrix;
        residual_multipliers = &constraint_rhs - &constraint_matrix * &solution;
        if gradient_descent.error_norm.apply(&residual_solution)
            < gradient_descent.abs_tol.residual()
            && gradient_descent.error_norm.apply(&residual_multipliers)
                < gradient_descent.abs_tol.constraint()
        {
            return Ok(solution);
        } else if steps == gradient_descent.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                gradient_descent.max_steps,
                format!("{gradient_descent:?}"),
            ));
        } else {
            steps += 1;
            step_size_solution = secant_solution.step_size(
                &gradient_descent.step_estimator,
                &solution,
                &residual_solution,
                step_size_solution,
            );
            step_size_multipliers = secant_multipliers.step_size(
                &gradient_descent.step_estimator,
                &multipliers,
                &residual_multipliers,
                step_size_multipliers,
            );
            step_size = step_size_solution.min(step_size_multipliers);
            solution -= residual_solution * StepSize::<F, X>::new(step_size);
            multipliers += residual_multipliers * step_size;
        }
    }
}

fn constrained_dual<S, Q, F, X, E>(
    gradient_descent: &GradientDescent<S, Q>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    S: ObjectiveMerit,
    Q: StepEstimator,
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if S::SEARCHES {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let num_constraints = constraint_rhs.len();
    let mut multipliers = Vector::zero(num_constraints);
    let mut secant = Secant::new(&multipliers);
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    for _ in 0..gradient_descent.max_steps {
        if let Ok(result) = unconstrained(
            gradient_descent,
            |_: &X| {
                panic!("Line search needs the exact penalty function in constrained optimization.")
            },
            &mut jacobian,
            solution.clone(),
            Some((&constraint_matrix, &multipliers)),
        ) {
            solution = result;
            residual = &constraint_rhs - &constraint_matrix * &solution;
            if gradient_descent.error_norm.apply(&residual) < gradient_descent.abs_tol.constraint()
            {
                return Ok(solution);
            } else {
                step_size = secant.step_size(
                    &gradient_descent.step_estimator,
                    &multipliers,
                    &residual,
                    step_size,
                );
                multipliers += residual * step_size;
            }
        } else {
            //
            // This sort of acts like LineSearch::Error, does it not?
            //
            multipliers -= (multipliers.clone() - secant.solution()) * CUTBACK_FACTOR_MINUS_ONE;
            step_size *= CUTBACK_FACTOR;
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        gradient_descent.max_steps,
        format!("{gradient_descent:?}"),
    ))
}
