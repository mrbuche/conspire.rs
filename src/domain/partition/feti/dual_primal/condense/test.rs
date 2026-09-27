use super::Condensed;
use crate::math::{SquareMatrix, Vector};

fn condense(
    local_stiffness: &SquareMatrix,
    local_force: &Vector,
    primal: &[usize],
    dual: &[usize],
) -> Condensed {
    Condensed::try_condense(local_stiffness, local_force, primal, dual)
        .expect("remainder block K_dd is singular")
}

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

#[test]
fn a_nearly_disconnected_dual_dof_is_refused_at_any_scale() {
    [1e-3, 1.0, 1e9, 1e12].into_iter().for_each(|scale| {
        let stiffness: SquareMatrix = [[4.0, 1.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 1e-13]]
            .into_iter()
            .map(|row| row.into_iter().map(|entry| entry * scale).collect())
            .collect();
        let force = Vector::zero(3);
        assert!(
            Condensed::try_condense(&stiffness, &force, &[0], &[1, 2]).is_none(),
            "scale {scale:e}: a nearly disconnected dof should be refused"
        );
    })
}
