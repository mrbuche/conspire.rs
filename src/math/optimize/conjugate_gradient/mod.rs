#[cfg(test)]
mod test;

use super::{
    super::{Jacobian, Matrix, Scalar, Solution, Tensor, Vector},
    EqualityConstraint, FirstOrderOptimization, LineSearch, LineSearcher, OptimizationError,
    StepSize, Tolerances, ZerothOrderRootFinding,
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

/// How much of the previous direction the next one inherits.
#[derive(Clone, Copy, Debug, Default)]
pub enum Conjugacy {
    /// Fletcher-Reeves.
    FletcherReeves,
    /// Hestenes-Stiefel.
    HestenesStiefel,
    /// Polak-Ribière.
    #[default]
    PolakRibiere,
}

impl Conjugacy {
    /// How much of the previous direction to carry, never less than none.
    ///
    /// Clamping at zero starts the direction over at steepest descent wherever
    /// the formula would carry the wrong way, which is what keeps the method
    /// honest without a schedule of restarts to keep alongside it.
    fn beta<E>(&self, residual: &E, previous: &E, direction: &E, change: &E) -> Scalar
    where
        E: Tensor,
    {
        let ratio = match self {
            Self::FletcherReeves => {
                residual.full_contraction(residual) / previous.full_contraction(previous)
            }
            Self::HestenesStiefel => {
                -residual.full_contraction(change) / direction.full_contraction(change)
            }
            Self::PolakRibiere => {
                residual.full_contraction(change) / previous.full_contraction(previous)
            }
        };
        if ratio.is_finite() {
            ratio.max(0.0)
        } else {
            0.0
        }
    }
}

/// The method of nonlinear conjugate gradients.
///
/// The direction is a decrement, so a step along it is subtracted rather than
/// added, and descent is the direction agreeing with the gradient.
///
/// Conjugacy is what the method has instead of curvature, and it holds exactly
/// when each step lands on the minimum along its direction. The step here is
/// estimated from the last two residuals instead, so conjugacy is approached
/// rather than held, and the clamp on the previous direction is what the method
/// leans on where it slips. That is why [`Conjugacy::PolakRibiere`] is the
/// default: it is the only one of the three whose formula can turn negative, so
/// it is the only one the clamp ever restarts.
///
/// [`LineSearch::Wolfe`] is what puts the curvature back, and the other two
/// formulas need it. [`Conjugacy::HestenesStiefel`] divides by the quantity the
/// curvature condition holds away from zero and does not converge without one.
pub struct ConjugateGradient {
    /// Absolute error tolerances.
    pub abs_tol: Tolerances,
    /// How much of the previous direction the next one inherits.
    pub conjugacy: Conjugacy,
    /// Lagrangian dual.
    pub dual: bool,
    /// Norm type for error evaluation.
    pub error_norm: Norm,
    /// Line search algorithm.
    pub line_search: LineSearch,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Relative error tolerance.
    pub rel_tol: Option<Scalar>,
}

impl<J, X> LineSearcher<J, X> for ConjugateGradient {
    fn get_line_search(&self) -> &LineSearch {
        &self.line_search
    }
}

impl Debug for ConjugateGradient {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ConjugateGradient {{ abs_tol: {:?}, conjugacy: {:?}, dual: {:?}, line_search: {}, max_steps: {:?}, rel_tol: {:?} }}",
            self.abs_tol, self.conjugacy, self.dual, self.line_search, self.max_steps, self.rel_tol
        )
    }
}

impl Default for ConjugateGradient {
    fn default() -> Self {
        Self {
            abs_tol: Tolerances::default(),
            conjugacy: Conjugacy::default(),
            dual: false,
            error_norm: Norm::Chebyshev,
            line_search: LineSearch::None,
            max_steps: 250,
            rel_tol: None,
        }
    }
}

impl<F, X, E> ZerothOrderRootFinding<F, X> for ConjugateGradient
where
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
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError> {
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => constrained_fixed(
                self,
                |_: &X| panic!("No line search in root finding."),
                function,
                initial_guess,
                indices,
            ),
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
            EqualityConstraint::None => unconstrained(
                self,
                |_: &X| panic!("No line search in root finding."),
                function,
                initial_guess,
                None,
            ),
        }
    }
}

impl<F, J, X, E> FirstOrderOptimization<F, J, X> for ConjugateGradient
where
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
        initial_guess: X,
        equality_constraint: EqualityConstraint,
    ) -> Result<X, OptimizationError> {
        let objective = move |argument: &X| function(argument).map(|value| *value.erase());
        match equality_constraint {
            EqualityConstraint::Fixed(indices) => {
                constrained_fixed(self, objective, jacobian, initial_guess, indices)
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

/// The direction the next step is taken along, and the size to try it at.
///
/// Both come from the same pair of differences, the conjugacy formula asking
/// how much of the previous direction still applies and the secant asking how
/// long a step that far has been worth.
fn conjugate<F, X, E>(
    conjugate_gradient: &ConjugateGradient,
    residual: &F,
    solution: &X,
    previous: Option<&(F, X, F)>,
    step_size: &mut Scalar,
) -> F
where
    F: Erase<Erased = E> + Tensor,
    X: Erase<Erased = E> + Tensor,
    E: Tensor,
{
    if let Some((residual_previous, solution_previous, direction_previous)) = previous {
        let change = residual.clone() - residual_previous;
        let step_trial = change
            .erase()
            .full_contraction((solution.clone() - solution_previous).erase())
            / change.erase().full_contraction(change.erase());
        if step_trial.abs() > 0.0 && !step_trial.is_nan() {
            *step_size = step_trial.abs()
        }
        let mut direction = direction_previous.clone()
            * conjugate_gradient.conjugacy.beta(
                residual.erase(),
                residual_previous.erase(),
                direction_previous.erase(),
                change.erase(),
            );
        direction += residual;
        if residual.erase().full_contraction(direction.erase()) > 0.0 {
            //
            // The secant asked how long a step along the gradient is worth, and
            // the step is about to be taken along the direction instead. Undoing
            // how much further the direction reaches keeps the step the distance
            // the secant meant it to be.
            //
            *step_size *= (residual.erase().full_contraction(residual.erase())
                / direction.erase().full_contraction(direction.erase()))
            .sqrt();
            return direction;
        }
    }
    residual.clone()
}

fn unconstrained<F, X, E>(
    conjugate_gradient: &ConjugateGradient,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    linear_equality_constraint: Option<(&Matrix, &Vector)>,
) -> Result<X, OptimizationError>
where
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
    let mut direction;
    let mut previous = None;
    let mut residual;
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    let mut steps = 0;
    loop {
        residual = if let Some(ref extra) = constraint {
            jacobian(&solution)? - extra
        } else {
            jacobian(&solution)?
        };
        if conjugate_gradient.error_norm.apply(&residual) < conjugate_gradient.abs_tol.residual() {
            return Ok(solution);
        } else if steps == conjugate_gradient.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                conjugate_gradient.max_steps,
                format!("{conjugate_gradient:?}"),
            ));
        } else {
            steps += 1;
            direction = conjugate(
                conjugate_gradient,
                &residual,
                &solution,
                previous.as_ref(),
                &mut step_size,
            );
            step_size = conjugate_gradient.search::<F, E>(
                |trial: &X, _: Scalar| function(trial),
                &mut jacobian,
                &solution,
                &residual,
                &direction,
                step_size,
            )?;
            previous = Some((residual, solution.clone(), direction.clone()));
            solution -= direction * StepSize::<F, X>::new(step_size);
        }
    }
}

fn constrained_fixed<F, X, E>(
    conjugate_gradient: &ConjugateGradient,
    mut function: impl FnMut(&X) -> Result<Scalar, String>,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    indices: Vec<usize>,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
{
    let mut direction;
    let mut previous = None;
    let mut relative_scale = 0.0;
    let mut residual: F;
    let mut residual_norm;
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    let mut steps = 0;
    loop {
        residual = jacobian(&solution)?;
        residual.zero_out(&indices);
        residual_norm = conjugate_gradient.error_norm.measure(&residual);
        if conjugate_gradient.rel_tol.is_some() && steps == 0 {
            relative_scale = residual_norm
        }
        if residual_norm < conjugate_gradient.abs_tol.residual {
            return Ok(solution);
        } else if let Some(rel_tol) = conjugate_gradient.rel_tol
            && residual_norm / relative_scale < rel_tol
        {
            return Ok(solution);
        } else if steps == conjugate_gradient.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                conjugate_gradient.max_steps,
                format!("{conjugate_gradient:?}"),
            ));
        } else {
            steps += 1;
            direction = conjugate(
                conjugate_gradient,
                &residual,
                &solution,
                previous.as_ref(),
                &mut step_size,
            );
            step_size = conjugate_gradient.search::<F, E>(
                |trial: &X, _: Scalar| function(trial),
                &mut jacobian,
                &solution,
                &residual,
                &direction,
                step_size,
            )?;
            previous = Some((residual, solution.clone(), direction.clone()));
            solution -= direction * StepSize::<F, X>::new(step_size);
        }
    }
}

/// Steps the variables and the multipliers at once.
///
/// The Lagrangian is descended in one and ascended in the other, so there is no
/// single objective for conjugacy to be conjugate against. The variables still
/// get a conjugate direction, the multipliers still only a gradient one, and
/// the step is the shorter of what each asks for.
fn constrained<F, X, E>(
    conjugate_gradient: &ConjugateGradient,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if !matches!(conjugate_gradient.line_search, LineSearch::None) {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let mut direction_solution;
    let mut previous = None;
    let mut residual_solution;
    let mut solution = initial_guess;
    let mut step_size_solution = INITIAL_STEP_SIZE;
    let num_constraints = constraint_rhs.len();
    let mut residual_multipliers;
    let mut residual_multipliers_change = Vector::zero(num_constraints);
    let mut multipliers = Vector::zero(num_constraints);
    let mut multipliers_change = Vector::zero(num_constraints);
    let mut step_size_multipliers = INITIAL_STEP_SIZE;
    let mut step_trial_multipliers;
    let mut step_size;
    let mut steps = 0;
    loop {
        residual_solution = jacobian(&solution)? - &multipliers * &constraint_matrix;
        residual_multipliers = &constraint_rhs - &constraint_matrix * &solution;
        if conjugate_gradient.error_norm.apply(&residual_solution)
            < conjugate_gradient.abs_tol.residual()
            && conjugate_gradient.error_norm.apply(&residual_multipliers)
                < conjugate_gradient.abs_tol.constraint()
        {
            return Ok(solution);
        } else if steps == conjugate_gradient.max_steps {
            return Err(OptimizationError::MaximumStepsReached(
                conjugate_gradient.max_steps,
                format!("{conjugate_gradient:?}"),
            ));
        } else {
            steps += 1;
            direction_solution = conjugate(
                conjugate_gradient,
                &residual_solution,
                &solution,
                previous.as_ref(),
                &mut step_size_solution,
            );
            multipliers_change -= &multipliers;
            residual_multipliers_change -= &residual_multipliers;
            step_trial_multipliers = residual_multipliers_change
                .full_contraction(&multipliers_change)
                / residual_multipliers_change.full_contraction(&residual_multipliers_change);
            if step_trial_multipliers.abs() > 0.0 && !step_trial_multipliers.is_nan() {
                step_size_multipliers = step_trial_multipliers.abs()
            }
            residual_multipliers_change = residual_multipliers.clone();
            multipliers_change = multipliers.clone();
            step_size = step_size_solution.min(step_size_multipliers);
            previous = Some((
                residual_solution,
                solution.clone(),
                direction_solution.clone(),
            ));
            solution -= direction_solution * StepSize::<F, X>::new(step_size);
            multipliers += residual_multipliers * step_size;
        }
    }
}

fn constrained_dual<F, X, E>(
    conjugate_gradient: &ConjugateGradient,
    mut jacobian: impl FnMut(&X) -> Result<F, String>,
    initial_guess: X,
    constraint_matrix: Matrix,
    constraint_rhs: Vector,
) -> Result<X, OptimizationError>
where
    F: Erase<Erased = E> + Jacobian + Mul<StepSize<F, X>, Output = X>,
    for<'a> &'a F: Mul<StepSize<F, X>, Output = X>,
    X: Erase<Erased = E> + Jacobian + Solution,
    <X as Tensor>::Unit: UnitDiv<<F as Tensor>::Unit>,
    E: Tensor,
    for<'a> &'a Matrix: Mul<&'a X, Output = Vector>,
{
    if !matches!(conjugate_gradient.line_search, LineSearch::None) {
        panic!("Line search needs the exact penalty function in constrained optimization.")
    }
    let num_constraints = constraint_rhs.len();
    let mut multipliers = Vector::zero(num_constraints);
    let mut multipliers_change = multipliers.clone();
    let mut residual;
    let mut residual_change = Vector::zero(num_constraints);
    let mut solution = initial_guess;
    let mut step_size = INITIAL_STEP_SIZE;
    let mut step_trial;
    for _ in 0..conjugate_gradient.max_steps {
        if let Ok(result) = unconstrained(
            conjugate_gradient,
            |_: &X| {
                panic!("Line search needs the exact penalty function in constrained optimization.")
            },
            &mut jacobian,
            solution.clone(),
            Some((&constraint_matrix, &multipliers)),
        ) {
            solution = result;
            residual = &constraint_rhs - &constraint_matrix * &solution;
            if conjugate_gradient.error_norm.apply(&residual)
                < conjugate_gradient.abs_tol.constraint()
            {
                return Ok(solution);
            } else {
                multipliers_change -= &multipliers;
                residual_change -= &residual;
                step_trial = residual_change.full_contraction(&multipliers_change)
                    / residual_change.full_contraction(&residual_change);
                if step_trial.abs() > 0.0 && !step_trial.is_nan() {
                    step_size = step_trial.abs()
                }
                residual_change = residual.clone();
                multipliers_change = multipliers.clone();
                multipliers += residual * step_size;
            }
        } else {
            multipliers -= (multipliers.clone() - &multipliers_change) * CUTBACK_FACTOR_MINUS_ONE;
            step_size *= CUTBACK_FACTOR;
        }
    }
    Err(OptimizationError::MaximumStepsReached(
        conjugate_gradient.max_steps,
        format!("{conjugate_gradient:?}"),
    ))
}
