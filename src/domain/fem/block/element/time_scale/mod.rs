#[cfg(test)]
mod test;

use crate::{
    domain::time_scale::{largest_eigenvalue as largest_eigenvalue_of, time_scale_from_eigenvalue},
    fem::block::element::{mass::ElementNodalLumpedMasses, solid::ElementNodalStiffnessesSolid},
    math::{Quantity, Scalar},
    units::Time,
};

/// The largest eigenvalue of $`M^{-1}K`$ for an element with the nodal stiffnesses
/// $`K`$ and the lumped masses $`M`$, by power iteration.
pub fn largest_eigenvalue<const N: usize, const P: usize>(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid<N>,
    nodal_lumped_masses: &ElementNodalLumpedMasses<P>,
) -> Scalar {
    assert_eq!(N, P, "The stiffnesses and masses must have the same nodes.");
    let masses: Vec<Scalar> = (0..3 * N)
        .map(|index| nodal_lumped_masses[index / 3].value())
        .collect();
    largest_eigenvalue_of(
        3 * N,
        |row, column| nodal_stiffnesses[row / 3][column / 3][row % 3][column % 3].value(),
        &masses,
    )
}

/// The fastest time scale of an element, the reciprocal of its highest angular frequency.
///
/// Infinite for an element that has no stiffness.
pub fn fastest_time_scale<const N: usize, const P: usize>(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid<N>,
    nodal_lumped_masses: &ElementNodalLumpedMasses<P>,
) -> Quantity<Time> {
    time_scale_from_eigenvalue(largest_eigenvalue(nodal_stiffnesses, nodal_lumped_masses))
}
