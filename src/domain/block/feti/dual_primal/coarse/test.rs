use super::{Coarse, CoarseSystem, assemble};
use crate::domain::block::feti::dual_primal::{
    BoundaryConditions, CornerDofs, CornerSelection, DualPrimalSplit, condense::Condensed,
};
use crate::math::{Matrix, SquareMatrix, Tensor};

fn one_by_one(value: f64) -> SquareMatrix {
    [[value]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect()
}

fn one_by_one_matrix(value: f64) -> Matrix {
    [[value]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect()
}

#[test]
fn shares_corner_contributions_across_subdomains() {
    let corners = CornerSelection::new(vec![42]);
    let corner_dofs = CornerDofs::new(&corners, &BoundaryConditions::none(), 1);
    let split_a = DualPrimalSplit::new(vec![0], vec![1]);
    let split_b = DualPrimalSplit::new(vec![0], vec![1]);
    let condensed_a = Condensed {
        schur: one_by_one(2.0),
        reduced_force: [1.0].into_iter().collect(),
        dual_map: one_by_one_matrix(0.0),
    };
    let condensed_b = Condensed {
        schur: one_by_one(3.0),
        reduced_force: [4.0].into_iter().collect(),
        dual_map: one_by_one_matrix(0.0),
    };
    let (schur, force) = assemble(
        &[condensed_a, condensed_b],
        &[split_a, split_b],
        &corner_dofs,
    );
    assert_eq!(schur.entry(0, 0), 5.0);
    assert_eq!(force[0], 5.0);
}

#[test]
fn solves_the_assembled_system() {
    let system = CoarseSystem {
        len: 1,
        pattern: vec![(0, 0)],
        values: vec![5.0],
    };
    let force: crate::math::Vector = [5.0].into_iter().collect();
    let solution = Coarse::new(system).solve(&force);
    assert!((solution[0] - 1.0).abs() < 1e-14);
}

/// Two overlapping local contributions to the same position must sum:
/// (0,0) gets 4 + 1, so the matrix is [[5, 1], [1, 3]] and [[5, 1], [1, 3]] . [1, 1] = [6, 4].
#[test]
fn sums_duplicate_positions_before_solving() {
    let system = CoarseSystem {
        len: 2,
        pattern: vec![(0, 0), (0, 1), (1, 0), (1, 1), (0, 0)],
        values: vec![4.0, 1.0, 1.0, 3.0, 1.0],
    };
    let force: crate::math::Vector = [6.0, 4.0].into_iter().collect();
    let solution = Coarse::new(system).solve(&force);
    assert!((solution[0] - 1.0).abs() < 1e-12);
    assert!((solution[1] - 1.0).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "singular")]
fn a_singular_coarse_problem_is_refused() {
    let system = CoarseSystem {
        len: 2,
        pattern: vec![(0, 0), (0, 1), (1, 0), (1, 1)],
        values: vec![1.0, 1.0, 1.0, 1.0],
    };
    Coarse::new(system);
}

#[test]
fn an_empty_coarse_problem_solves_to_nothing() {
    let system = CoarseSystem {
        len: 0,
        pattern: Vec::new(),
        values: Vec::new(),
    };
    let coarse = Coarse::new(system);
    assert_eq!(coarse.len(), 0);
    assert_eq!(coarse.solve(&crate::math::Vector::zero(0)).len(), 0);
}
