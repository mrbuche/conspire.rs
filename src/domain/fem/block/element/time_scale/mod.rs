#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{mass::ElementNodalLumpedMasses, solid::ElementNodalStiffnessesSolid},
    math::{Quantity, Scalar},
    units::Time,
};

const MAXIMUM_ITERATIONS: usize = 500;
const TOLERANCE: Scalar = 1e-10;

/// The largest eigenvalue of $`M^{-1}K`$, by power iteration, for an element
/// with the nodal stiffnesses $`K`$ and the lumped masses $`M`$.
///
/// Iterates from the same deterministic, non-symmetric start every time, until
/// the Rayleigh quotient changes by less than a relative tolerance. The
/// estimate approaches the largest eigenvalue from below, so it can fall short
/// of it by about the tolerance, or by more where the largest eigenvalues are
/// clustered and the iteration stops early.
pub fn largest_eigenvalue<const N: usize, const P: usize>(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid<N>,
    nodal_lumped_masses: &ElementNodalLumpedMasses<P>,
) -> Scalar {
    assert_eq!(N, P, "The stiffnesses and masses must have the same nodes.");
    let size = 3 * N;
    let stiffness = |row: usize, column: usize| {
        nodal_stiffnesses[row / 3][column / 3][row % 3][column % 3].value()
    };
    let masses: Vec<Scalar> = (0..size)
        .map(|index| nodal_lumped_masses[index / 3].value())
        .collect();
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
        let mass: Scalar = vector.iter().zip(&masses).map(|(v, m)| v * v * m).sum();
        let previous = eigenvalue;
        eigenvalue = energy / mass;
        let next: Vec<Scalar> = product.iter().zip(&masses).map(|(p, m)| p / m).collect();
        let norm = next
            .iter()
            .zip(&masses)
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

/// The fastest time scale of an element, the reciprocal of its highest angular frequency.
///
/// Infinite for an element that has no stiffness.
pub fn fastest_time_scale<const N: usize, const P: usize>(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid<N>,
    nodal_lumped_masses: &ElementNodalLumpedMasses<P>,
) -> Quantity<Time> {
    let eigenvalue = largest_eigenvalue(nodal_stiffnesses, nodal_lumped_masses);
    if eigenvalue > 0.0 {
        Time::seconds(1.0 / eigenvalue.sqrt())
    } else {
        Time::seconds(Scalar::INFINITY)
    }
}
