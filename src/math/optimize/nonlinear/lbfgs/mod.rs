#[cfg(test)]
mod test;

use crate::math::{
    ContractWith, FlatView, Quantity, Scalar, Solution, Tensor, Vector,
    optimize::{
        Armijo, EqualityConstraint, EqualityConstraints, ObjectiveMerit, Optimization,
        OptimizationError, Tolerances,
    },
    sparse::SparseSolver,
};
use crate::units::{Dimensionless, UnitMul};
use std::{collections::VecDeque, mem::swap};

const CURVATURE_FLOOR: Scalar = 1.0e-12;

/// The limited-memory Broyden-Fletcher-Goldfarb-Shanno method.[^1]
///
/// [^1]: D.C. Liu and J. Nocedal, [Math. Program. **45**, 503 (1989)](https://doi.org/10.1007/BF01589116).
#[derive(Clone, Debug)]
pub struct LimitedMemoryBFGS<S = Armijo> {
    /// Absolute error tolerances, the residual being the length of the gradient.
    pub abs_tol: Tolerances,
    /// Number of corrections kept.
    pub history: usize,
    /// Line search algorithm.
    pub line_search: S,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Relative error tolerance, on the length of the gradient at the initial guess.
    pub rel_tol: Option<Scalar>,
    /// Length of the first step, in the unit of the solution.
    pub typical: Scalar,
}

/// The corrections a descent has kept, which a later descent on a nearby function can start from.
#[derive(Clone, Debug, Default)]
pub struct Corrections(VecDeque<(Vector, Vector, Scalar)>);

/// The outcome of a descent, which may end before the tolerances are met.
#[derive(Clone, Debug)]
pub struct Descent<F, X> {
    /// Whether the gradient met the tolerances.
    pub converged: bool,
    /// The function at the initial guess.
    pub initial: F,
    /// The final iterate.
    pub solution: X,
    /// The number of steps taken.
    pub steps: usize,
    /// The function at the final iterate.
    pub value: F,
}

impl Default for LimitedMemoryBFGS<Armijo> {
    fn default() -> Self {
        Self {
            abs_tol: Tolerances::default(),
            history: 10,
            line_search: Armijo {
                control: 1.0e-4,
                cut_back: 0.5,
                max_steps: 40,
            },
            max_steps: 250,
            rel_tol: None,
            typical: 1.0,
        }
    }
}

impl<S, F, J, X, U> Optimization<F, J, (), X> for LimitedMemoryBFGS<S>
where
    S: ObjectiveMerit,
    F: Clone + Into<Quantity<U>> + Tensor,
    J: ContractWith<X, Output = Quantity<U>> + FlatView + Tensor,
    X: FlatView + Solution + Tensor,
    Dimensionless: UnitMul<U, Output = U>,
{
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        _hessian: impl FnMut(&X) -> Result<(), String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
        _sparse: Option<SparseSolver>,
    ) -> Result<X, OptimizationError> {
        let descent = self.descend(function, jacobian, initial_guess, equality_constraint)?;
        if descent.converged {
            Ok(descent.solution)
        } else if descent.steps < self.max_steps {
            Err(OptimizationError::Intermediate(format!(
                "No step along the descent direction decreases the function, in solver: {self:?}."
            )))
        } else {
            Err(OptimizationError::MaximumStepsReached(
                self.max_steps,
                format!("{self:?}"),
            ))
        }
    }
}

impl<S> LimitedMemoryBFGS<S> {
    /// The same solver, searching along each direction with another line search.
    pub fn with_line_search<T>(self, line_search: T) -> LimitedMemoryBFGS<T> {
        LimitedMemoryBFGS {
            abs_tol: self.abs_tol,
            history: self.history,
            line_search,
            max_steps: self.max_steps,
            rel_tol: self.rel_tol,
            typical: self.typical,
        }
    }
}

impl<S> LimitedMemoryBFGS<S>
where
    S: ObjectiveMerit,
{
    /// Descends until the tolerances are met, no step decreases the function,
    /// or the step budget is spent, returning where it got to in any case.
    pub fn descend<F, J, X, U>(
        &self,
        function: impl FnMut(&X) -> Result<F, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
    ) -> Result<Descent<F, X>, OptimizationError>
    where
        F: Clone + Into<Quantity<U>> + Tensor,
        J: ContractWith<X, Output = Quantity<U>> + FlatView + Tensor,
        X: FlatView + Solution + Tensor,
        Dimensionless: UnitMul<U, Output = U>,
    {
        self.descend_from(
            &mut Corrections::default(),
            function,
            jacobian,
            initial_guess,
            equality_constraint,
        )
    }
    /// Descends as [`descend`](Self::descend) does, starting from the corrections kept by an
    /// earlier descent and leaving the ones it keeps in their place.
    pub fn descend_from<F, J, X, U>(
        &self,
        corrections: &mut Corrections,
        mut function: impl FnMut(&X) -> Result<F, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
        mut initial_guess: X,
        equality_constraint: EqualityConstraints,
    ) -> Result<Descent<F, X>, OptimizationError>
    where
        F: Clone + Into<Quantity<U>> + Tensor,
        J: ContractWith<X, Output = Quantity<U>> + FlatView + Tensor,
        X: FlatView + Solution + Tensor,
        Dimensionless: UnitMul<U, Output = U>,
    {
        let unsupported = || {
            OptimizationError::Intermediate(format!(
                "These equality constraints are not supported, in solver: {self:?}."
            ))
        };
        let Some(equality_constraint) =
            EqualityConstraint::lower_single(equality_constraint, &mut initial_guess)
        else {
            return Err(unsupported());
        };
        let fixed = match equality_constraint {
            EqualityConstraint::Fixed(indices) => indices,
            EqualityConstraint::None => Vec::new(),
            EqualityConstraint::Prescribed(_) => {
                unreachable!("Prescribed constraints are lowered first.")
            }
            EqualityConstraint::Multipliers(_) | EqualityConstraint::Linear(..) => {
                return Err(unsupported());
            }
        };
        let mut gradient = |solution: &X| -> Result<J, String> {
            let mut residual = jacobian(solution)?;
            fixed
                .iter()
                .for_each(|&index| residual.as_flat_mut()[index] = 0.0);
            Ok(residual)
        };
        let n = initial_guess.as_flat().len();
        let mut solution = initial_guess;
        let mut trial = solution.clone();
        let mut residual = gradient(&solution)?;
        let mut direction = Vector::zero(n);
        let mut alphas = Vec::with_capacity(self.history);
        let history = &mut corrections.0;
        if history
            .front()
            .is_some_and(|(s, _, _)| s.as_slice().len() != n)
        {
            history.clear();
        }
        let initial = function(&solution)?;
        let mut value = initial.clone();
        let mut reference = None;
        let mut steps = 0;
        let mut converged = false;
        while steps < self.max_steps {
            let magnitude = norm(residual.as_flat());
            let scale = *reference.get_or_insert(magnitude);
            if magnitude < self.abs_tol.residual
                || self
                    .rel_tol
                    .is_some_and(|rel_tol| magnitude < rel_tol * scale)
            {
                converged = true;
                break;
            }
            steps += 1;
            two_loop(
                residual.as_flat(),
                &*history,
                self.typical / magnitude,
                direction.as_mut_slice(),
                &mut alphas,
            );
            let slope = dot(residual.as_flat(), direction.as_slice());
            if slope >= 0.0 {
                history.clear();
                continue;
            }
            let decrement: X = direction
                .iter()
                .map(|direction| -direction)
                .collect::<Vector>()
                .into();
            let known: Quantity<U> = value.clone().into();
            let initial_slope = residual.contract_with(&decrement);
            let mut accepted = None;
            let mut merit = |step: Quantity<Dimensionless>| {
                if step.value() == 0.0 {
                    return Ok(known);
                }
                let step = step.value();
                trial
                    .as_flat_mut()
                    .iter_mut()
                    .zip(solution.as_flat().iter().zip(&direction))
                    .for_each(|(trial, (solution, direction))| {
                        *trial = solution + direction * step
                    });
                let candidate = function(&trial)?;
                let merit: Quantity<U> = candidate.clone().into();
                accepted = Some((candidate, step));
                Ok(merit)
            };
            let searched = self
                .line_search
                .search(
                    &mut merit,
                    |step: Quantity<Dimensionless>| {
                        if step.value() == 0.0 {
                            return Ok(initial_slope);
                        }
                        let mut point = solution.clone();
                        point
                            .as_flat_mut()
                            .iter_mut()
                            .zip(solution.as_flat().iter().zip(&direction))
                            .for_each(|(point, (solution, direction))| {
                                *point = solution + direction * step.value()
                            });
                        Ok(gradient(&point)?.contract_with(&decrement))
                    },
                    Dimensionless::of(1.0),
                )
                .map_err(|error| OptimizationError::upstream(error, self))
                .and_then(|step| {
                    if S::SEARCHES {
                        Ok(())
                    } else {
                        merit(step).map(drop).map_err(OptimizationError::from)
                    }
                });
            let Some((candidate, step)) = accepted.filter(|_| searched.is_ok()) else {
                if history.is_empty() {
                    break;
                }
                history.clear();
                continue;
            };
            swap(&mut solution, &mut trial);
            let updated = gradient(&solution)?;
            let (mut s, mut y) = if history.len() == self.history {
                let (s, y, _) = history.pop_front().unwrap();
                (s, y)
            } else {
                (Vector::zero(n), Vector::zero(n))
            };
            s.iter_mut()
                .zip(direction.iter())
                .for_each(|(s, direction)| *s = direction * step);
            y.iter_mut()
                .zip(updated.as_flat().iter().zip(residual.as_flat()))
                .for_each(|(y, (updated, residual))| *y = updated - residual);
            let curvature = dot(s.as_slice(), y.as_slice());
            if curvature > CURVATURE_FLOOR * norm(s.as_slice()) * norm(y.as_slice()) {
                history.push_back((s, y, 1.0 / curvature));
            }
            residual = updated;
            value = candidate;
        }
        Ok(Descent {
            converged,
            initial,
            solution,
            steps,
            value,
        })
    }
}

fn two_loop(
    gradient: &[Scalar],
    history: &VecDeque<(Vector, Vector, Scalar)>,
    fallback: Scalar,
    q: &mut [Scalar],
    alphas: &mut Vec<Scalar>,
) {
    q.iter_mut()
        .zip(gradient.iter())
        .for_each(|(q, gradient)| *q = *gradient);
    alphas.clear();
    history.iter().rev().for_each(|(s, y, rho)| {
        let alpha = rho * dot(s.as_slice(), q);
        q.iter_mut()
            .zip(y.iter())
            .for_each(|(q, y)| *q -= alpha * y);
        alphas.push(alpha)
    });
    let gamma = history.back().map_or(fallback, |(s, y, _)| {
        dot(s.as_slice(), y.as_slice()) / dot(y.as_slice(), y.as_slice())
    });
    q.iter_mut().for_each(|q| *q *= -gamma);
    history
        .iter()
        .zip(alphas.iter().rev())
        .for_each(|((s, y, rho), alpha)| {
            let beta = rho * dot(y.as_slice(), q);
            q.iter_mut()
                .zip(s.iter())
                .for_each(|(q, s)| *q -= (alpha + beta) * s)
        });
}

fn dot(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.algebraic_mul(*b))
        .fold(0.0, Scalar::algebraic_add)
}

fn norm(a: &[Scalar]) -> Scalar {
    dot(a, a).sqrt()
}
