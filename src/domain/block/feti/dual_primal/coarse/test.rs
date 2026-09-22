use super::{assemble, solve};
use crate::domain::block::feti::dual_primal::{
    CornerSelection, DualPrimalSplit, condense::Condensed,
};
use crate::math::{Matrix, SquareMatrix};

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
        &corners,
        1,
    );
    assert_eq!(schur[0][0], 5.0);
    assert_eq!(force[0], 5.0);
}

#[test]
fn solves_the_assembled_system() {
    let schur = one_by_one(5.0);
    let force: crate::math::Vector = [5.0].into_iter().collect();
    let solution = solve(&schur, &force);
    assert_eq!(solution[0], 1.0);
}
