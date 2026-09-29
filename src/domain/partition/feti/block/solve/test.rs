use super::super::super::{dual_primal::BoundaryConditions, pcg::Preconditioner};
use super::{SolveError, solve_local_systems};
use crate::{
    geometry::mesh::Partition,
    math::{SquareMatrix, Vector, optimize::KrylovMethod},
};

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

#[test]
fn a_subdomain_with_a_singular_interior_is_refused_though_its_dual_block_is_not() {
    let partition = Partition::from_parts_nodes(vec![vec![0, 1, 2, 3, 4, 5], vec![3, 4, 6, 7, 8]]);
    let mut first = identity(18);
    (0..3).for_each(|i| {
        first[12 + i][12 + i] = 0.0;
        first[15 + i][15 + i] = 0.0;
        first[12 + i][15 + i] = 1.0;
        first[15 + i][12 + i] = 1.0;
    });
    let second = identity(15);
    let positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 2.0, 2.0],
        [3.0, 2.0, 2.0],
        [2.0, 3.0, 2.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let pinned = [0, 1, 2, 6, 7, 8];
    let result = solve_local_systems(
        &partition,
        &BoundaryConditions::new(
            pinned
                .iter()
                .flat_map(|&node| (0..3).map(move |component| (node, component)))
                .collect(),
        ),
        vec![first, second],
        vec![Vector::zero(18), Vector::zero(15)],
        &positions,
        Preconditioner::Dirichlet,
        1e-8,
        KrylovMethod::default(),
    );
    assert!(matches!(result, Err(SolveError::SingularInterior(0))));
}
