#[cfg(test)]
pub use crate::fem::block::element::quadratic::tetrahedron::test::{
    D as TETRAHEDRON_D, applied_velocities as tetrahedron_applied_velocities,
    applied_velocity as tetrahedron_applied_velocity,
    equality_constraint as tetrahedron_equality_constraint,
    get_connectivity as tetrahedron_get_connectivity,
    get_coordinates_block as tetrahedron_get_coordinates_block,
    get_reference_coordinates_block as tetrahedron_get_reference_coordinates_block,
    get_velocities_block as tetrahedron_get_velocities_block,
    reference_coordinates as tetrahedron_reference_coordinates,
};
