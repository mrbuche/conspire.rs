use crate::{
    domain::time_scale::{largest_eigenvalue as largest_eigenvalue_of, time_scale_from_eigenvalue},
    math::{Quantity, Scalar, Tensor},
    units::Time,
    vem::block::element::{mass::ElementNodalLumpedMasses, solid::ElementNodalStiffnessesSolid},
};

/// The largest eigenvalue of $`M^{-1}K`$ for an element with the nodal stiffnesses
/// $`K`$ and the lumped masses $`M`$, by power iteration.
pub fn largest_eigenvalue(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid,
    nodal_lumped_masses: &ElementNodalLumpedMasses,
) -> Scalar {
    assert_eq!(
        nodal_stiffnesses.len(),
        nodal_lumped_masses.len(),
        "The stiffnesses and masses must have the same nodes."
    );
    let masses: Vec<Scalar> = (0..3 * nodal_lumped_masses.len())
        .map(|index| nodal_lumped_masses[index / 3].value())
        .collect();
    largest_eigenvalue_of(
        masses.len(),
        |row, column| nodal_stiffnesses[row / 3][column / 3][row % 3][column % 3].value(),
        &masses,
    )
}

/// The fastest time scale of an element, the reciprocal of its highest angular frequency.
///
/// Infinite for an element that has no stiffness.
pub fn fastest_time_scale(
    nodal_stiffnesses: &ElementNodalStiffnessesSolid,
    nodal_lumped_masses: &ElementNodalLumpedMasses,
) -> Quantity<Time> {
    time_scale_from_eigenvalue(largest_eigenvalue(nodal_stiffnesses, nodal_lumped_masses))
}
