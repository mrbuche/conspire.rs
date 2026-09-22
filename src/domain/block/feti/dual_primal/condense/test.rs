use super::condense;
use crate::domain::block::feti::dual_primal::DualPrimalSplit;
use crate::math::{SquareMatrix, Vector};

#[test]
fn schur_complement_and_reduced_force() {
    let stiffness: SquareMatrix = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let force: Vector = [1.0, 2.0, 3.0].into_iter().collect();
    let split = DualPrimalSplit::new(vec![1], vec![0, 2]);
    let condensed = condense(&stiffness, &force, &split);
    assert_eq!(condensed.schur[0][0], 3.75);
    assert_eq!(condensed.schur[0][1], -0.25);
    assert_eq!(condensed.schur[1][0], -0.25);
    assert_eq!(condensed.schur[1][1], 3.75);
    assert_eq!(condensed.reduced_force[0], 0.5);
    assert_eq!(condensed.reduced_force[1], 2.5);
    assert_eq!(condensed.primal_map[0][0], 0.25);
    assert_eq!(condensed.primal_map[0][1], 0.25);
}
