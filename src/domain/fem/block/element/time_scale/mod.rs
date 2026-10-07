#[cfg(test)]
mod test;

use crate::{
    domain::time_scale::{
        largest_eigenvalue as largest_eigenvalue_of, time_scale_exceeds as time_scale_exceeds_of,
        time_scale_from_eigenvalue,
    },
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

/// Whether the fastest time scale of an element is certified to exceed `minimum`.
///
/// Certified by the inertia of an LDLᵀ factorization, so unlike [`fastest_time_scale`], which
/// approaches the highest frequency from below, it cannot pass an element that is too fast.
/// A time scale equal to `minimum`, up to rounding, is not certified.
pub fn time_scale_exceeds<const N: usize, const P: usize>(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid<N>,
    nodal_lumped_masses: &ElementNodalLumpedMasses<P>,
    minimum: Quantity<Time>,
) -> bool {
    assert_eq!(N, P, "The stiffnesses and masses must have the same nodes.");
    let masses: Vec<Scalar> = (0..3 * N)
        .map(|index| nodal_lumped_masses[index / 3].value())
        .collect();
    time_scale_exceeds_of(
        3 * N,
        |row, column| nodal_stiffnesses[row / 3][column / 3][row % 3][column % 3].value(),
        &masses,
        minimum,
    )
}
