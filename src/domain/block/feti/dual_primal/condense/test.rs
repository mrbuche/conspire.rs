use super::condense;
use crate::math::{SquareMatrix, Vector};

#[test]
fn schur_complement_and_reduced_force() {
    let stiffness: SquareMatrix = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let force: Vector = [1.0, 2.0, 3.0].into_iter().collect();
    let condensed = condense(&stiffness, &force, &[1], &[0, 2]);
    assert_eq!(condensed.schur[0][0], 3.5);
    assert_eq!(condensed.reduced_force[0], 1.0);
    assert_eq!(condensed.dual_map[0][0], 0.25);
    assert_eq!(condensed.dual_map[1][0], 0.25);
}
