use super::super::super::dual_primal::BoundaryConditions;
use super::{SolveError, solve};
use crate::{
    constitutive::solid::{
        elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
        hyperelastic::NeoHookean,
    },
    fem::{
        NodalCoordinates, NodalReferenceCoordinates,
        block::{Block, element::linear::Tetrahedron},
    },
    geometry::mesh::Partition,
    math::Tensor,
};

fn coordinates() -> Vec<[f64; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 2.0],
    ]
}

fn block() -> Block<NeoHookean, Tetrahedron, 1, 3, 4, 4> {
    let reference_coordinates = NodalReferenceCoordinates::from(coordinates());
    Block::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        vec![[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 5]],
        &reference_coordinates,
    ))
}

fn partition() -> Partition {
    Partition::from_parts_nodes(vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![1, 2, 3, 5]])
}

#[test]
fn a_free_floating_assembly_has_no_boundary_condition_to_pin_it() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let result = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &BoundaryConditions::none(),
        3,
    );
    assert!(matches!(result, Err(SolveError::SingularCoarseProblem)));
}

#[test]
fn a_supported_assembly_at_zero_deformation_solves_to_zero_displacement() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let boundary_conditions = BoundaryConditions::new(vec![
        (0, 0),
        (0, 1),
        (0, 2),
        (4, 0),
        (4, 1),
        (4, 2),
        (5, 0),
        (5, 1),
        (5, 2),
    ]);
    let solution = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &boundary_conditions,
        3,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}

#[test]
fn a_boundary_condition_on_a_corner_node_solves_correctly() {
    let block = block();
    let nodal_coordinates = NodalCoordinates::from(coordinates());
    let boundary_conditions =
        BoundaryConditions::new(vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)]);
    let solution = solve(
        &block,
        &nodal_coordinates,
        &partition(),
        &boundary_conditions,
        3,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}
