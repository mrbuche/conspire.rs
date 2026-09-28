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

#[test]
fn the_primal_map_is_the_transposed_dual_map_for_a_symmetric_tangent() {
    let stiffness: SquareMatrix = [[4.0, 1.0, 0.5], [1.0, 4.0, 1.0], [0.5, 1.0, 4.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let force = Vector::zero(3);
    let condensed = condense(&stiffness, &force, &[1], &[0, 2]);
    (0..2).for_each(|d| {
        assert!((condensed.primal_map[0][d] - condensed.dual_map[d][0]).abs() < 1e-14)
    });
}

#[test]
fn the_primal_map_is_k_pd_times_the_inverse_of_k_dd_for_a_nonsymmetric_tangent() {
    let stiffness: SquareMatrix = [
        [4.0, 1.0, -2.0, 0.5],
        [3.0, 5.0, 0.7, -1.0],
        [-1.0, 2.0, 6.0, 1.5],
        [0.5, -0.3, 2.0, 4.0],
    ]
    .into_iter()
    .map(|row| row.into_iter().collect())
    .collect();
    let force = Vector::zero(4);
    let (primal, dual) = ([1, 3], [0, 2]);
    let condensed = condense(&stiffness, &force, &primal, &dual);
    let k_dd: SquareMatrix = dual
        .iter()
        .map(|&row| dual.iter().map(|&col| stiffness[row][col]).collect())
        .collect();
    let inverse_columns: Vec<Vector> = (0..2)
        .map(|d| {
            let unit: Vector = (0..2).map(|i| if i == d { 1.0 } else { 0.0 }).collect();
            k_dd.solve_lu(&unit).unwrap()
        })
        .collect();
    primal.iter().enumerate().for_each(|(p, &primal_dof)| {
        dual.iter().enumerate().for_each(|(d, _)| {
            let expected: f64 = dual
                .iter()
                .enumerate()
                .map(|(e, &dual_dof)| stiffness[primal_dof][dual_dof] * inverse_columns[d][e])
                .sum();
            assert!((condensed.primal_map[p][d] - expected).abs() < 1e-12);
        })
    });
    assert!((condensed.primal_map[0][0] - condensed.dual_map[0][0]).abs() > 1e-3);
}
