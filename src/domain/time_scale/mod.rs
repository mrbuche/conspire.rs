#[cfg(test)]
mod test;

use crate::{
    math::{Quantity, Scalar, SquareMatrix},
    units::Time,
};

const MAXIMUM_ITERATIONS: usize = 500;
const TOLERANCE: Scalar = 1e-10;

/// The largest eigenvalue of $`M^{-1}K`$, by power iteration, for the stiffness
/// $`K`$ of `size` degrees of freedom and the lumped masses $`M`$ on them.
///
/// Iterates from the same deterministic, non-symmetric start every time, until
/// the Rayleigh quotient changes by less than a relative tolerance. The
/// estimate approaches the largest eigenvalue from below, so it can fall short
/// of it by about the tolerance, or by more where the largest eigenvalues are
/// clustered and the iteration stops early.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn largest_eigenvalue(
    size: usize,
    stiffness: impl Fn(usize, usize) -> Scalar,
    masses: &[Scalar],
) -> Scalar {
    assert_eq!(
        size,
        masses.len(),
        "There must be a mass for each degree of freedom."
    );
    let mut vector: Vec<Scalar> = (0..size)
        .map(|index| ((index as u64 * 2654435761) % 4294967296) as Scalar / 4294967296.0 - 0.5)
        .collect();
    let mut eigenvalue = 0.0;
    for _ in 0..MAXIMUM_ITERATIONS {
        let product: Vec<Scalar> = (0..size)
            .map(|row| {
                (0..size)
                    .map(|column| stiffness(row, column) * vector[column])
                    .sum()
            })
            .collect();
        let energy: Scalar = vector.iter().zip(&product).map(|(v, p)| v * p).sum();
        let mass: Scalar = vector.iter().zip(masses).map(|(v, m)| v * v * m).sum();
        let previous = eigenvalue;
        eigenvalue = energy / mass;
        let next: Vec<Scalar> = product.iter().zip(masses).map(|(p, m)| p / m).collect();
        let norm = next
            .iter()
            .zip(masses)
            .map(|(w, m)| w * w * m)
            .sum::<Scalar>()
            .sqrt();
        if norm == 0.0 {
            return 0.0;
        }
        vector = next.into_iter().map(|w| w / norm).collect();
        if (eigenvalue - previous).abs() <= TOLERANCE * eigenvalue.abs() {
            break;
        }
    }
    eigenvalue
}

#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn time_scale_from_eigenvalue(eigenvalue: Scalar) -> Quantity<Time> {
    if eigenvalue > 0.0 {
        Time::seconds(1.0 / eigenvalue.sqrt())
    } else {
        Time::seconds(Scalar::INFINITY)
    }
}

/// The time scale of a diffusive eigenvalue, the reciprocal of the largest eigenvalue of
/// $`C^{-1}K`$, which bounds a stable explicit time step.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn diffusive_time_scale_from_eigenvalue(eigenvalue: Scalar) -> Quantity<Time> {
    if eigenvalue > 0.0 {
        Time::seconds(1.0 / eigenvalue)
    } else {
        Time::seconds(Scalar::INFINITY)
    }
}

/// Whether every eigenvalue of $`M^{-1}K`$ is below `bound`, for the stiffness $`K`$ of `size`
/// degrees of freedom and the lumped masses $`M`$ on them.
///
/// Certified by the inertia of $`\text{bound}\,M - K`$, which is positive definite exactly then.
/// Unlike [`largest_eigenvalue`] it cannot fall short of the largest eigenvalue, but a bound that
/// is equal to it, up to rounding, is not certified.
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn eigenvalues_below(
    size: usize,
    stiffness: impl Fn(usize, usize) -> Scalar,
    masses: &[Scalar],
    bound: Scalar,
) -> bool {
    assert_eq!(
        size,
        masses.len(),
        "There must be a mass for each degree of freedom."
    );
    let mut matrix = SquareMatrix::zero(size);
    (0..size).for_each(|row| {
        (0..=row).for_each(|column| {
            let entry = -0.5 * (stiffness(row, column) + stiffness(column, row));
            matrix[row][column] = entry;
            matrix[column][row] = entry
        });
        matrix[row][row] += bound * masses[row]
    });
    matrix
        .factorize_ldl()
        .is_ok_and(|decomposition| decomposition.inertia() == (size, 0, 0))
}

/// Whether the fastest time scale is certified to exceed `minimum`, see [`eigenvalues_below`].
#[cfg_attr(not(any(feature = "fem", feature = "vem")), allow(dead_code))]
pub(crate) fn time_scale_exceeds(
    size: usize,
    stiffness: impl Fn(usize, usize) -> Scalar,
    masses: &[Scalar],
    minimum: Quantity<Time>,
) -> bool {
    eigenvalues_below(
        size,
        stiffness,
        masses,
        1.0 / (minimum.value() * minimum.value()),
    )
}
