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
use crate::units::{Dimensionless, UnitDiv, UnitMul};
use std::fmt::{self, Debug, Formatter};
use std::{
    collections::VecDeque,
    mem::swap,
    ops::Range,
    sync::{
        Barrier,
        atomic::{AtomicU64, Ordering},
    },
    thread::{available_parallelism, scope},
};

const BLOCK: usize = 4096;
const GRAIN: usize = 3 * BLOCK;
const CURVATURE_FLOOR: Scalar = 1.0e-12;

/// The limited-memory Broyden-Fletcher-Goldfarb-Shanno method.[^1]
///
/// [^1]: D.C. Liu and J. Nocedal, [Math. Program. **45**, 503 (1989)](https://doi.org/10.1007/BF01589116).
#[derive(Clone)]
pub struct LimitedMemoryBFGS<S = Armijo, W = Dimensionless> {
    /// Absolute error tolerances, the residual being the length of the gradient.
    pub abs_tol: Tolerances,
    /// Number of corrections kept.
    pub history: usize,
    /// Line search algorithm.
    pub line_search: S,
    /// Maximum number of steps.
    pub max_steps: usize,
    /// Relative error tolerance, on the length of the gradient at the initial guess.
    pub rel_tol: Option<Quantity>,
    /// Length of the first step, in the unit of the solution.
    pub typical: Quantity<W>,
}

impl<S, W> Debug for LimitedMemoryBFGS<S, W>
where
    S: Debug,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LimitedMemoryBFGS {{ abs_tol: {:?}, history: {:?}, line_search: {:?}, max_steps: {:?}, rel_tol: {:?}, typical: {:?} }}",
            self.abs_tol,
            self.history,
            self.line_search,
            self.max_steps,
            self.rel_tol,
            self.typical
        )
    }
}

/// The corrections a descent has kept, which a later descent on a nearby function can start from.
#[derive(Clone, Debug, Default)]
pub struct Corrections(VecDeque<(Vector, Vector, Scalar)>);

impl Corrections {
    /// The entries that are not zero in some kept correction.
    pub fn nonzero(&self) -> Vec<bool> {
        let mut nonzero = vec![false; self.0.front().map_or(0, |(s, _, _)| s.as_slice().len())];
        self.0.iter().for_each(|(s, y, _)| {
            [s, y].into_iter().for_each(|vector| {
                nonzero
                    .iter_mut()
                    .zip(vector.iter())
                    .for_each(|(nonzero, &entry)| *nonzero |= entry != 0.0)
            })
        });
        nonzero
    }
    /// Moves the entries of the kept corrections to the positions `map` gives, in vectors of
    /// `length` entries that are zero elsewhere. Entries that `map` sends to `usize::MAX` must
    /// be zero, as the entries of a descent confined to the others are.
    pub fn remap(&mut self, map: &[usize], length: usize) {
        self.0.iter_mut().for_each(|(s, y, _)| {
            [s, y].into_iter().for_each(|vector| {
                let mut moved = Vector::zero(length);
                map.iter()
                    .zip(vector.iter())
                    .filter(|(_, entry)| **entry != 0.0)
                    .for_each(|(&to, &entry)| moved.as_mut_slice()[to] = entry);
                *vector = moved
            })
        })
    }
}

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

impl<W> Default for LimitedMemoryBFGS<Armijo, W> {
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
            typical: Quantity::new(1.0),
        }
    }
}

impl<S, W, J, X, U> Optimization<Quantity<U>, J, (), X> for LimitedMemoryBFGS<S, W>
where
    S: ObjectiveMerit,
    J: ContractWith<X, Output = Quantity<U>> + FlatView + Tensor,
    X: FlatView + Solution + Tensor<Unit = W>,
    Dimensionless: UnitMul<U, Output = U>,
    W: UnitDiv<<J as Tensor>::Unit>,
{
    fn minimize(
        &self,
        function: impl FnMut(&X) -> Result<Quantity<U>, String>,
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

impl<S, W> LimitedMemoryBFGS<S, W> {
    /// The same solver, searching along each direction with another line search.
    pub fn with_line_search<T>(self, line_search: T) -> LimitedMemoryBFGS<T, W> {
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

impl<S, W> LimitedMemoryBFGS<S, W>
where
    S: ObjectiveMerit,
{
    /// Descends until the tolerances are met, no step decreases the function,
    /// or the step budget is spent, returning where it got to in any case.
    pub fn descend<J, X, U>(
        &self,
        function: impl FnMut(&X) -> Result<Quantity<U>, String>,
        jacobian: impl FnMut(&X) -> Result<J, String>,
        initial_guess: X,
        equality_constraint: EqualityConstraints,
    ) -> Result<Descent<Quantity<U>, X>, OptimizationError>
    where
        J: ContractWith<X, Output = Quantity<U>> + FlatView + Tensor,
        X: FlatView + Solution + Tensor<Unit = W>,
        Dimensionless: UnitMul<U, Output = U>,
        W: UnitDiv<<J as Tensor>::Unit>,
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
    pub fn descend_from<J, X, U>(
        &self,
        corrections: &mut Corrections,
        mut function: impl FnMut(&X) -> Result<Quantity<U>, String>,
        mut jacobian: impl FnMut(&X) -> Result<J, String>,
        mut initial_guess: X,
        equality_constraint: EqualityConstraints,
    ) -> Result<Descent<Quantity<U>, X>, OptimizationError>
    where
        J: ContractWith<X, Output = Quantity<U>> + FlatView + Tensor,
        X: FlatView + Solution + Tensor<Unit = W>,
        Dimensionless: UnitMul<U, Output = U>,
        W: UnitDiv<<J as Tensor>::Unit>,
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
        let history = &mut corrections.0;
        if history
            .front()
            .is_some_and(|(s, _, _)| s.as_slice().len() != n)
        {
            history.clear();
        }
        let initial = function(&solution)?;
        let mut value = initial;
        let mut reference = None;
        let mut steps = 0;
        let mut converged = false;
        while steps < self.max_steps {
            let magnitude = residual.norm();
            let scale = *reference.get_or_insert(magnitude);
            if magnitude < self.abs_tol.residual()
                || self
                    .rel_tol
                    .is_some_and(|rel_tol| magnitude.ratio(scale) < rel_tol)
            {
                converged = true;
                break;
            }
            steps += 1;
            two_loop(
                residual.as_flat(),
                &*history,
                (self.typical / magnitude).value(),
                direction.as_mut_slice(),
            );
            let decrement: X = direction
                .iter()
                .map(|direction| -direction)
                .collect::<Vector>()
                .into();
            let initial_slope = residual.contract_with(&decrement);
            if initial_slope <= Quantity::default() {
                history.clear();
                continue;
            }
            let mut accepted = None;
            let mut merit = |step: Quantity<Dimensionless>| {
                if step.value() == 0.0 {
                    return Ok(value);
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
                accepted = Some((candidate, step));
                Ok(candidate)
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

struct Reduction {
    barrier: Barrier,
    slots: [Vec<AtomicU64>; 2],
}

impl Reduction {
    fn new(workers: usize, blocks: usize) -> Self {
        Self {
            barrier: Barrier::new(workers),
            slots: [(); 2].map(|()| (0..blocks).map(|_| AtomicU64::new(0)).collect()),
        }
    }
    fn sum(&self, turn: &mut usize, mine: Range<usize>, block: impl Fn(usize) -> Scalar) -> Scalar {
        let slots = &self.slots[*turn % 2];
        *turn += 1;
        mine.for_each(|index| slots[index].store(block(index).to_bits(), Ordering::Relaxed));
        self.barrier.wait();
        slots
            .iter()
            .map(|slot| Scalar::from_bits(slot.load(Ordering::Relaxed)))
            .fold(0.0, Scalar::algebraic_add)
    }
}

fn view(vector: &Vector, offset: usize, length: usize) -> &[Scalar] {
    &vector.as_slice()[offset..offset + length]
}

fn two_loop(
    gradient: &[Scalar],
    history: &VecDeque<(Vector, Vector, Scalar)>,
    fallback: Scalar,
    q: &mut [Scalar],
) {
    let threads = available_parallelism().map_or(1, |threads| threads.get());
    two_loop_with(
        (q.len() / GRAIN).clamp(1, threads),
        gradient,
        history,
        fallback,
        q,
    )
}

fn two_loop_with(
    workers: usize,
    gradient: &[Scalar],
    history: &VecDeque<(Vector, Vector, Scalar)>,
    fallback: Scalar,
    q: &mut [Scalar],
) {
    q.iter_mut()
        .zip(gradient.iter())
        .for_each(|(q, gradient)| *q = *gradient);
    if q.is_empty() {
        return;
    }
    let blocks = q.len().div_ceil(BLOCK);
    let per = blocks.div_ceil(workers.clamp(1, blocks));
    let reduction = Reduction::new(blocks.div_ceil(per), blocks);
    let recursion = |worker: usize, q: &mut [Scalar]| {
        let first = worker * per;
        let offset = first * BLOCK;
        let length = q.len();
        let mine = first..first + length.div_ceil(BLOCK);
        let span = |index: usize| {
            let low = (index - first) * BLOCK;
            (low, (low + BLOCK).min(length))
        };
        let mut turn = 0;
        let mut alphas = Vec::with_capacity(history.len());
        for (s, y, rho) in history.iter().rev() {
            let s = view(s, offset, length);
            let alpha = rho
                * reduction.sum(&mut turn, mine.clone(), |index| {
                    let (low, high) = span(index);
                    partial(&s[low..high], &q[low..high])
                });
            q.iter_mut()
                .zip(view(y, offset, length))
                .for_each(|(q, y)| *q -= alpha * y);
            alphas.push(alpha)
        }
        let gamma = match history.back() {
            Some((s, y, _)) => {
                let (s, y) = (view(s, offset, length), view(y, offset, length));
                let sy = reduction.sum(&mut turn, mine.clone(), |index| {
                    let (low, high) = span(index);
                    partial(&s[low..high], &y[low..high])
                });
                let yy = reduction.sum(&mut turn, mine.clone(), |index| {
                    let (low, high) = span(index);
                    partial(&y[low..high], &y[low..high])
                });
                sy / yy
            }
            None => fallback,
        };
        q.iter_mut().for_each(|q| *q *= -gamma);
        for ((s, y, rho), alpha) in history.iter().zip(alphas.iter().rev()) {
            let y = view(y, offset, length);
            let beta = rho
                * reduction.sum(&mut turn, mine.clone(), |index| {
                    let (low, high) = span(index);
                    partial(&y[low..high], &q[low..high])
                });
            q.iter_mut()
                .zip(view(s, offset, length))
                .for_each(|(q, s)| *q -= (alpha + beta) * s)
        }
    };
    if blocks.div_ceil(per) == 1 {
        recursion(0, q)
    } else {
        scope(|scope| {
            q.chunks_mut(per * BLOCK)
                .enumerate()
                .for_each(|(worker, q)| {
                    let recursion = &recursion;
                    scope.spawn(move || recursion(worker, q));
                })
        });
    }
}

fn partial(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| a.algebraic_mul(*b))
        .fold(0.0, Scalar::algebraic_add)
}

fn dot(a: &[Scalar], b: &[Scalar]) -> Scalar {
    a.chunks(BLOCK)
        .zip(b.chunks(BLOCK))
        .map(|(a, b)| partial(a, b))
        .fold(0.0, Scalar::algebraic_add)
}

fn norm(a: &[Scalar]) -> Scalar {
    dot(a, a).sqrt()
}
