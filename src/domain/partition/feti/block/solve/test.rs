use super::super::super::{dual_primal::BoundaryConditions, pcg::Preconditioner};
use super::{SolveError, solve, solve_local_systems};
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
    math::{SquareMatrix, Tensor, Vector, optimize::KrylovMethod},
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
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    assert_eq!(solution.len(), 6 * 3);
    solution.iter().for_each(|&entry| {
        assert!(entry.is_finite());
        assert!(entry.abs() < 1e-8);
    });
}

fn identity(len: usize) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(len);
    (0..len).for_each(|i| matrix[i][i] = 1.0);
    matrix
}

#[test]
fn a_subdomain_with_no_corners_and_no_boundary_conditions_is_refused_as_floating() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2], vec![1, 2, 3]]);
    let local_stiffnesses = vec![identity(9), identity(9)];
    let local_forces = vec![Vector::zero(9), Vector::zero(9)];
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let result = solve_local_systems(
        &partition,
        &BoundaryConditions::none(),
        local_stiffnesses,
        local_forces,
        &positions,
        Preconditioner::Dirichlet,
        1e-8,
        KrylovMethod::default(),
    );
    assert!(matches!(
        result,
        Err(SolveError::FloatingSubdomain {
            part: 0,
            removed: 0
        })
    ));
}
