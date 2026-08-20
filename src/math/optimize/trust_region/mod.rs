#[cfg(test)]
mod test;

use super::super::{Norm, Scalar, SquareMatrix, Tensor, Vector};

/// How far a step is trusted to follow the model it was built from.
#[derive(Clone, Copy, Debug, Default)]
pub enum TrustRegion {
    /// The step is shortened to a radius that never adapts.
    Fixed { radius: Scalar, norm: Norm },
    /// The step is taken whole, however far the model carries it.
    #[default]
    None,
}

const MORE_SORENSEN_TOLERANCE: Scalar = 1e-6;
const MORE_SORENSEN_MAX_ITERATIONS: usize = 100;

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

/// Solves `tangent` shifted by `lambda`, panicking if that shift is singular.
///
/// Only ever called at a `lambda` already known (from `more_sorensen`'s own
/// search) to factorize, so the panic is unreachable in practice.
fn step_at(
    tangent: &SquareMatrix,
    residual: &Vector,
    lambda: Scalar,
    num_variables: usize,
) -> Vector {
    shift(tangent, lambda, num_variables)
        .factorize_ldl()
        .expect("admissible shift found by more_sorensen's own search")
        .solve(residual)
}

/// Solves the trust-region subproblem for a (possibly bordered/KKT) symmetric
/// tangent, using inertia rather than mere factorization success to know when
/// a shift is admissible, since a saddle-point matrix can never be positive
/// definite by construction.
///
/// `num_variables` is the size of the primal block that gets shifted; the
/// remaining `tangent.len() - num_variables` rows are multiplier rows expected
/// to contribute exactly one negative eigenvalue each. Passing the full size
/// (`num_variables == tangent.len()`) degenerates to the plain unconstrained
/// case, expecting the whole tangent to become positive definite.
///
/// The true Moré-Sorensen hard case (a shift landing exactly on the positive
/// definite boundary with room to spare inside the radius) is not handled;
/// the boundary solution is returned as-is.
#[allow(dead_code)]
pub(crate) fn more_sorensen(
    tangent: &SquareMatrix,
    residual: &Vector,
    radius: Scalar,
    num_variables: usize,
) -> Vector {
    let num_constraints = tangent.len() - num_variables;
    let expected = (num_variables, num_constraints, 0);
    if let Ok(decomposition) = tangent.factorize_ldl()
        && decomposition.inertia() == expected
    {
        let step = decomposition.solve(residual);
        if step_norm(&step, num_variables) <= radius {
            return step;
        }
    }
    let scale = (0..num_variables)
        .map(|i| tangent[i][i].abs())
        .fold(1.0, Scalar::max);
    let mut lambda = scale;
    let mut lambda_bar = None;
    for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
        if shift(tangent, lambda, num_variables)
            .factorize_ldl()
            .is_ok_and(|decomposition| decomposition.inertia() == expected)
        {
            lambda_bar = Some(lambda);
            break;
        }
        lambda *= 2.0
    }
    let lambda_bar = lambda_bar.expect("no admissible shift found within the iteration budget");
    let step_bar = step_at(tangent, residual, lambda_bar, num_variables);
    if step_norm(&step_bar, num_variables) <= radius {
        return step_bar;
    }
    let (mut lambda_lo, mut lambda_hi) = (lambda_bar, lambda_bar * 2.0);
    while step_norm(
        &step_at(tangent, residual, lambda_hi, num_variables),
        num_variables,
    ) > radius
    {
        lambda_hi *= 2.0
    }
    let mut step = step_at(tangent, residual, lambda_hi, num_variables);
    for _ in 0..MORE_SORENSEN_MAX_ITERATIONS {
        let lambda_mid = 0.5 * (lambda_lo + lambda_hi);
        step = step_at(tangent, residual, lambda_mid, num_variables);
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
