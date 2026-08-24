#[cfg(test)]
mod test;

use super::super::{Norm, Scalar, SquareMatrix, Tensor, Vector, sparse::SparseSolver};

/// How far a step is trusted to follow the model it was built from.
#[derive(Clone, Copy, Debug, Default)]
pub enum TrustRegion {
    /// A radius that grows and shrinks with how well the model predicted the
    /// step it was asked for, accepting or rejecting each step in turn.
    ///
    /// Only honored on the dense, symmetric KKT path (`minimize`, not root
    /// finding); when active it supersedes whatever `line_search` is set to,
    /// the two being alternative globalization strategies rather than
    /// composable ones.
    Adaptive { radius: Scalar, max_radius: Scalar },
    /// The step is shortened to a radius that never adapts.
    Fixed { radius: Scalar, norm: Norm },
    /// The step is taken whole, however far the model carries it.
    #[default]
    None,
}

/// The reduction ratio above which a trust-region step is accepted.
pub(crate) const TRUST_REGION_ETA: Scalar = 0.1;
const TRUST_REGION_SHRINK_THRESHOLD: Scalar = 0.25;
const TRUST_REGION_GROW_THRESHOLD: Scalar = 0.75;
const TRUST_REGION_SHRINK: Scalar = 0.25;
const TRUST_REGION_GROW: Scalar = 2.0;
pub(crate) const TRUST_REGION_MAX_REJECTIONS: usize = 25;

// How precisely the bisection locates the radius boundary. The radius is a
// heuristic that only ever gets doubled or quartered, so resolving its boundary
// to more than a couple of digits buys nothing and costs a factorization per
// digit. `update_radius` has to read the boundary at least this loosely too: a
// step is only converged to within this of the radius, so a tighter test there
// would never see a step as having reached the boundary and the radius would
// never grow.
const MORE_SORENSEN_TOLERANCE: Scalar = 1e-2;
// How far a shift is nudged to step off an exact singularity, which is a
// question about floating-point spacing rather than about the radius, and stays
// tight however loosely the boundary itself is located.
const MORE_SORENSEN_NUDGE: Scalar = 1e-6;
const MORE_SORENSEN_MAX_ITERATIONS: usize = 100;
// How far a shift is let grow relative to the tangent's own scale before
// giving up on shrinking further: past this, the arithmetic used to test a
// shift for singularity loses so much precision that a whole neighborhood of
// lambda reads as singular, not just an isolated point, and no amount of
// nudging escapes that (this is what an unbounded search ran into in
// practice, chasing an infeasible radius toward lambda ~ 1e12).
const MORE_SORENSEN_LAMBDA_CAP: Scalar = 1e8;

/// Updates the trust-region radius from the reduction ratio and how close the
/// step that ratio was measured against came to the radius boundary
/// (Nocedal & Wright, Algorithm 4.1).
pub(crate) fn update_radius(
    radius: Scalar,
    max_radius: Scalar,
    rho: Scalar,
    primal_step_norm: Scalar,
) -> Scalar {
    if rho < TRUST_REGION_SHRINK_THRESHOLD {
        TRUST_REGION_SHRINK * primal_step_norm
    } else if rho > TRUST_REGION_GROW_THRESHOLD
        && primal_step_norm >= radius * (1.0 - MORE_SORENSEN_TOLERANCE)
    {
        (TRUST_REGION_GROW * radius).min(max_radius)
    } else {
        radius
    }
}

/// The tangent shifted by `lambda` on only the primal (first `num_variables`) rows,
/// the multiplier rows of a bordered KKT block left untouched.
fn shift(tangent: &SquareMatrix, lambda: Scalar, num_variables: usize) -> SquareMatrix {
    let mut shifted = tangent.clone();
    (0..num_variables).for_each(|i| shifted[i][i] += lambda);
    shifted
}

/// The Euclidean norm of only the primal block of a step, the multiplier rows
/// being of another kind entirely and not part of what the radius bounds.
fn step_norm(step: &Vector, num_variables: usize) -> Scalar {
    Norm::Euclidean.over(step.iter().take(num_variables).copied())
}

/// Solves the shifted system, nudging past a `lambda` landing exactly on a
/// singular shift (only `lambda_bar` and the doubling bracket's endpoint are
/// individually inertia-checked; a bisection midpoint can still land on an
/// exact singularity for structured matrices, even though the admissible
/// region on either side of it is generically open).
fn step_at(
    solve: &mut impl FnMut(Scalar) -> Option<(Vector, (usize, usize, usize))>,
    lambda: Scalar,
) -> Vector {
    let mut lambda = lambda;
    for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
        if let Some((step, _)) = solve(lambda) {
            return step;
        }
        lambda *= 1.0 + MORE_SORENSEN_NUDGE
    }
    panic!("no non-singular shift found near lambda = {lambda}")
}

/// Solves the trust-region subproblem for a (possibly bordered/KKT) symmetric
/// system, using inertia rather than mere factorization success to know when a
/// shift is admissible, since a saddle-point matrix can never be positive
/// definite by construction.
///
/// `solve` is handed a shift and returns the step the shifted system takes
/// along with its inertia, or nothing if that shift cannot be factorized; it is
/// what keeps this free of how the system is stored, a dense tangent and a
/// sparse pattern differing only in how they answer it. `scale` is the largest
/// magnitude on the primal diagonal, which is what a shift is measured
/// against, and `size` the extent of the whole bordered system.
///
/// `num_variables` is the size of the primal block that gets shifted; the
/// remaining `size - num_variables` rows are multiplier rows expected to
/// contribute exactly one negative eigenvalue each. Passing the full size
/// (`num_variables == size`) degenerates to the plain unconstrained case,
/// expecting the whole system to become positive definite.
///
/// The true Moré-Sorensen hard case (a shift landing exactly on the positive
/// definite boundary with room to spare inside the radius) is not handled;
/// the boundary solution is returned as-is.
///
/// The equality-constraint rows are never shifted, so a shift can only ever
/// shrink the step so far: the constraint forces the primal step onto an
/// affine set whose minimum-norm point is a floor no lambda goes below. A
/// `radius` beneath that floor returns the smallest step found instead of
/// hitting the radius exactly.
pub(crate) fn more_sorensen(
    size: usize,
    scale: Scalar,
    mut solve: impl FnMut(Scalar) -> Option<(Vector, (usize, usize, usize))>,
    radius: Scalar,
    num_variables: usize,
) -> Vector {
    let expected = (num_variables, size - num_variables, 0);
    let at_zero = solve(0.0);
    //
    // Shifting the primal block of an already-admissible system by a positive
    // lambda keeps it admissible, the reduced Hessian only gaining a positive
    // definite term, so admissibility here means the whole interval above zero
    // is admissible and the step norm falls monotonically across it.
    //
    let admissible_at_zero = matches!(&at_zero, Some((_, inertia)) if *inertia == expected);
    if let Some((step, _)) = at_zero
        && admissible_at_zero
        && step_norm(&step, num_variables) <= radius
    {
        return step;
    }
    let cap = scale * MORE_SORENSEN_LAMBDA_CAP;
    let mut lambda = scale;
    let mut admissible = None;
    while lambda <= cap {
        if let Some((step, inertia)) = solve(lambda)
            && inertia == expected
        {
            admissible = Some((lambda, step));
            break;
        }
        lambda *= 2.0
    }
    // No lambda up to the cap made the shift admissible at all: the model
    // itself (not just the constraint floor below) can't be regularized into
    // a descent direction within the precision this scale supports. Fall
    // back to the unregularized Newton direction rather than fail outright.
    let Some((lambda_bar, step_bar)) = admissible else {
        return solve(0.0)
            .map(|(step, _)| step)
            .unwrap_or_else(|| Vector::zero(size));
    };
    let fits = step_norm(&step_bar, num_variables) <= radius;
    if fits && !admissible_at_zero {
        // Nothing below `lambda_bar` is admissible at all, so a step already
        // inside the radius is as close to the boundary as this model reaches.
        return step_bar;
    }
    let (mut lambda_lo, mut lambda_hi, step_hi) = if fits {
        //
        // The doubling search only looks for an admissible shift, and starts
        // at the tangent's own scale; when zero was admissible already that
        // overshoots the boundary badly, damping a step that only needed
        // shortening. Bracket back down toward the unshifted step instead of
        // taking the overshoot.
        //
        (0.0, lambda_bar, step_bar)
    } else {
        // The equality-constraint rows are never shifted, so they force the
        // primal step to satisfy `constraint_matrix * step = constraint_residual`
        // exactly regardless of lambda: the minimum-norm point on that affine
        // set is a floor on the achievable step norm no shift can go below. If
        // doubling stalls (or the cap is hit first), the requested radius is
        // beneath that floor; return the smallest step found rather than
        // chasing an unreachable target toward ever larger lambda.
        let mut lambda_hi = lambda_bar * 2.0;
        let mut step_hi = step_at(&mut solve, lambda_hi);
        while step_norm(&step_hi, num_variables) > radius && lambda_hi < cap {
            lambda_hi = (lambda_hi * 2.0).min(cap);
            step_hi = step_at(&mut solve, lambda_hi);
        }
        if step_norm(&step_hi, num_variables) > radius {
            return step_hi;
        }
        (lambda_bar, lambda_hi, step_hi)
    };
    let mut step = step_hi;
    for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
        let lambda_mid = 0.5 * (lambda_lo + lambda_hi);
        step = step_at(&mut solve, lambda_mid);
        let norm = step_norm(&step, num_variables);
        if (norm - radius).abs() <= MORE_SORENSEN_TOLERANCE * radius {
            break;
        } else if norm > radius {
            lambda_lo = lambda_mid
        } else {
            lambda_hi = lambda_mid
        }
    }
    step
}

/// The trust-region subproblem for a dense tangent held whole.
pub(crate) fn more_sorensen_dense(
    tangent: &SquareMatrix,
    residual: &Vector,
    radius: Scalar,
    num_variables: usize,
) -> Vector {
    more_sorensen(
        tangent.len(),
        (0..num_variables)
            .map(|i| tangent[i][i].abs())
            .fold(1.0, Scalar::max),
        |lambda| {
            shift(tangent, lambda, num_variables)
                .factorize_ldl()
                .ok()
                .map(|decomposition| (decomposition.solve(residual), decomposition.inertia()))
        },
        radius,
        num_variables,
    )
}

/// The trust-region subproblem for a system given as entries on a sparse
/// pattern, the shift riding along in the source rather than in a matrix of
/// its own: the pattern already holds every primal diagonal, so shifting one
/// changes no structure and every trial reuses the same symbolic
/// factorization.
pub(crate) fn more_sorensen_sparse(
    solver: &SparseSolver,
    entry: impl Fn(usize, usize) -> Scalar,
    residual: &Vector,
    radius: Scalar,
    num_variables: usize,
) -> Vector {
    more_sorensen(
        residual.len(),
        (0..num_variables)
            .map(|i| entry(i, i).abs())
            .fold(1.0, Scalar::max),
        |lambda| {
            solver
                .solve_ldl(
                    |i, j| {
                        entry(i, j)
                            + if i == j && i < num_variables {
                                lambda
                            } else {
                                0.0
                            }
                    },
                    residual,
                )
                .ok()
        },
        radius,
        num_variables,
    )
}

/// The quadratic form of only the primal block of a bordered tangent, the
/// multiplier rows having no part in what the model predicts.
pub(crate) fn quadratic_dense(
    tangent: &SquareMatrix,
    step: &Vector,
    num_variables: usize,
) -> Scalar {
    (0..num_variables)
        .map(|i| {
            (0..num_variables)
                .map(|j| step[i] * tangent[i][j] * step[j])
                .sum::<Scalar>()
        })
        .sum()
}
