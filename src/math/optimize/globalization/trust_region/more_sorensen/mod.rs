#[cfg(test)]
mod test;

use std::convert::Infallible;

use crate::math::{
    Norm, Scalar, SquareMatrix, Tensor, Vector,
    sparse::{SparseError, SparseSolver},
};

pub(super) const MORE_SORENSEN_TOLERANCE: Scalar = 1e-2;
const MORE_SORENSEN_NUDGE: Scalar = 1e-6;
const MORE_SORENSEN_MAX_ITERATIONS: usize = 100;
const MORE_SORENSEN_LAMBDA_CAP: Scalar = 1e8;

fn shift(tangent: &SquareMatrix, lambda: Scalar, num_variables: usize) -> SquareMatrix {
    let mut shifted = tangent.clone();
    (0..num_variables).for_each(|i| shifted[i][i] += lambda);
    shifted
}

fn step_norm(step: &Vector, num_variables: usize) -> Scalar {
    Norm::Euclidean.over(step.iter().take(num_variables).copied())
}

fn step_at<E>(
    solve: &mut impl FnMut(Scalar) -> Result<Option<(Vector, (usize, usize, usize))>, E>,
    lambda: Scalar,
) -> Result<Vector, E> {
    let mut lambda = lambda;
    for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
        if let Some((step, _)) = solve(lambda)? {
            return Ok(step);
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
/// along with its inertia, or nothing if that shift cannot be factorized, or an
/// error that no shift could cure, which ends the search at once; it is
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
pub(crate) fn more_sorensen<E>(
    size: usize,
    scale: Scalar,
    mut solve: impl FnMut(Scalar) -> Result<Option<(Vector, (usize, usize, usize))>, E>,
    radius: Scalar,
    num_variables: usize,
) -> Result<Vector, E> {
    let expected = (num_variables, size - num_variables, 0);
    let at_zero = solve(0.0)?;
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
        return Ok(step);
    }
    let cap = scale * MORE_SORENSEN_LAMBDA_CAP;
    let mut lambda = scale;
    let mut below = 0.0;
    let mut admissible = None;
    while lambda <= cap {
        if let Some((step, inertia)) = solve(lambda)?
            && inertia == expected
        {
            admissible = Some((lambda, step));
            break;
        }
        below = lambda;
        lambda *= 2.0
    }
    // No lambda up to the cap made the shift admissible at all: the model
    // itself (not just the constraint floor below) can't be regularized into
    // a descent direction within the precision this scale supports. Fall
    // back to the unregularized Newton direction rather than fail outright.
    let Some((lambda_bar, step_bar)) = admissible else {
        return Ok(solve(0.0)?
            .map(|(step, _)| step)
            .unwrap_or_else(|| Vector::zero(size)));
    };
    let (mut lambda_lo, mut lambda_hi, mut step_hi) = if admissible_at_zero {
        // The doubling search starts at the tangent's own scale, which
        // overshoots a boundary that only needed the step shortened; bracket
        // back down toward the unshifted step instead.
        (0.0, lambda_bar, step_bar)
    } else {
        //
        // The same overshoot afflicts the first admissible shift when zero is
        // not: it is only the first one doubling reached, and the smallest
        // admissible shift can lie orders of magnitude below the tangent's
        // scale. Admissibility is monotone in lambda, so bisect for its edge.
        //
        let (mut lo, mut hi, mut step) = (below, lambda_bar, step_bar);
        for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
            if hi - lo <= MORE_SORENSEN_TOLERANCE * hi {
                break;
            }
            let mid = 0.5 * (lo + hi);
            match solve(mid)? {
                Some((mid_step, inertia)) if inertia == expected => {
                    hi = mid;
                    step = mid_step
                }
                _ => lo = mid,
            }
        }
        if step_norm(&step, num_variables) <= radius {
            return Ok(step);
        }
        (hi, 2.0 * hi, step_at(&mut solve, 2.0 * hi)?)
    };
    // The equality-constraint rows are never shifted, so they force the
    // primal step to satisfy `constraint_matrix * step = constraint_residual`
    // exactly regardless of lambda: the minimum-norm point on that affine
    // set is a floor on the achievable step norm no shift can go below. If
    // doubling stalls (or the cap is hit first), the requested radius is
    // beneath that floor; return the smallest step found rather than
    // chasing an unreachable target toward ever larger lambda.
    while step_norm(&step_hi, num_variables) > radius && lambda_hi < cap {
        lambda_lo = lambda_hi;
        lambda_hi = (lambda_hi * 2.0).min(cap);
        step_hi = step_at(&mut solve, lambda_hi)?;
    }
    if step_norm(&step_hi, num_variables) > radius {
        return Ok(step_hi);
    }
    let mut step = step_hi;
    for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
        let lambda_mid = 0.5 * (lambda_lo + lambda_hi);
        step = step_at(&mut solve, lambda_mid)?;
        let norm = step_norm(&step, num_variables);
        if (norm - radius).abs() <= MORE_SORENSEN_TOLERANCE * radius {
            break;
        } else if norm > radius {
            lambda_lo = lambda_mid
        } else {
            lambda_hi = lambda_mid
        }
    }
    Ok(step)
}

pub(crate) fn more_sorensen_dense(
    tangent: &SquareMatrix,
    residual: &Vector,
    radius: Scalar,
    num_variables: usize,
) -> Vector {
    more_sorensen::<Infallible>(
        tangent.len(),
        (0..num_variables)
            .map(|i| tangent[i][i].abs())
            .fold(1.0, Scalar::max),
        |lambda| {
            Ok(shift(tangent, lambda, num_variables)
                .factorize_ldl()
                .ok()
                .map(|decomposition| (decomposition.solve(residual), decomposition.inertia())))
        },
        radius,
        num_variables,
    )
    .unwrap_or_else(|never| match never {})
}

pub(crate) fn more_sorensen_sparse(
    solver: &SparseSolver,
    entry: impl Fn(usize, usize) -> Scalar,
    residual: &Vector,
    radius: Scalar,
    num_variables: usize,
) -> Result<Vector, SparseError> {
    more_sorensen(
        residual.len(),
        (0..num_variables)
            .map(|i| entry(i, i).abs())
            .fold(1.0, Scalar::max),
        |lambda| match solver.solve_ldl(
            |i, j| {
                entry(i, j)
                    + if i == j && i < num_variables {
                        lambda
                    } else {
                        0.0
                    }
            },
            residual,
        ) {
            Err(SparseError::Unsymmetric) => Err(SparseError::Unsymmetric),
            result => Ok(result.ok()),
        },
        radius,
        num_variables,
    )
}
