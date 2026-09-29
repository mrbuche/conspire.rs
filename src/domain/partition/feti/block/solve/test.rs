use super::super::super::{Formulation, dual_primal::BoundaryConditions, pcg::Preconditioner};
use super::{SolveError, solve_local_systems};
use crate::{
    geometry::mesh::Partition,
    math::{SquareMatrix, Tensor, Vector, optimize::KrylovMethod},
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
        Formulation::DualPrimal,
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
        Formulation::DualPrimal,
    );
    assert!(matches!(result, Err(SolveError::SingularInterior(0))));
}

const TRUSS: [[f64; 3]; 5] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
];

fn bars(nodes: &[usize]) -> SquareMatrix {
    let mut stiffness = SquareMatrix::zero(3 * nodes.len());
    (0..nodes.len()).for_each(|a| {
        ((a + 1)..nodes.len()).for_each(|b| {
            let mut direction = [0.0; 3];
            (0..3).for_each(|axis| direction[axis] = TRUSS[nodes[b]][axis] - TRUSS[nodes[a]][axis]);
            let length = direction.iter().map(|d| d * d).sum::<f64>().sqrt();
            let stretch = 1.0 + a as f64 + 2.0 * b as f64;
            (0..3).for_each(|i| {
                (0..3).for_each(|j| {
                    let value = stretch * direction[i] * direction[j] / (length * length);
                    stiffness[3 * a + i][3 * a + j] += value;
                    stiffness[3 * b + i][3 * b + j] += value;
                    stiffness[3 * a + i][3 * b + j] -= value;
                    stiffness[3 * b + i][3 * a + j] -= value;
                })
            })
        })
    });
    stiffness
}

#[test]
fn classical_reproduces_a_dense_solve_of_the_assembled_truss() {
    let parts = vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4]];
    let partition = Partition::from_parts_nodes(parts.clone());
    let stiffnesses: Vec<SquareMatrix> = parts.iter().map(|nodes| bars(nodes)).collect();
    let forces: Vec<Vector> = parts
        .iter()
        .enumerate()
        .map(|(part, nodes)| {
            (0..3 * nodes.len())
                .map(|dof| ((dof + 3 * part) as f64 * 0.7).sin() + 0.3)
                .collect()
        })
        .collect();
    let mut pinned: Vec<(usize, usize)> = (0..3).map(|c| (0, c)).collect();
    pinned.extend((0..3).map(|c| (4, c)));
    pinned.push((1, 1));
    let free: Vec<usize> = (0..15)
        .filter(|&dof| !pinned.contains(&(dof / 3, dof % 3)))
        .collect();
    let mut global = SquareMatrix::zero(15);
    let mut load = Vector::zero(15);
    parts.iter().enumerate().for_each(|(part, nodes)| {
        (0..3 * nodes.len()).for_each(|i| {
            load[3 * nodes[i / 3] + i % 3] += forces[part][i];
            (0..3 * nodes.len()).for_each(|j| {
                global[3 * nodes[i / 3] + i % 3][3 * nodes[j / 3] + j % 3] +=
                    stiffnesses[part][i][j]
            })
        })
    });
    let reduced: SquareMatrix = free
        .iter()
        .map(|&row| free.iter().map(|&col| global[row][col]).collect())
        .collect();
    let expected = reduced
        .factorize_lu()
        .unwrap()
        .solve(&free.iter().map(|&dof| load[dof]).collect());
    let (solution, _) = solve_local_systems(
        &partition,
        &BoundaryConditions::new(pinned),
        stiffnesses,
        forces,
        &TRUSS,
        Preconditioner::Dirichlet,
        1e-12,
        KrylovMethod::default(),
        Formulation::Classical,
    )
    .unwrap_or_else(|error| panic!("{error}"));
    free.iter().zip(expected.iter()).for_each(|(&dof, &value)| {
        assert!(
            (solution[dof] - value).abs() < 1e-8 * (1.0 + value.abs()),
            "dof {dof}: {} against {value}",
            solution[dof]
        )
    });
}
