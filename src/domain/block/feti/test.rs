use super::{Subdomain, dual_action, projected_pcg};
use crate::domain::block::feti::{
    dual_primal::{CornerSelection, build_splits},
    interface::{Partition, build_interfaces},
};
use crate::math::{SquareMatrix, Vector};

/// A two-node bar of stiffness `k`, corner at local index 0, dual DOF at 1.
fn stiffness(k: f64) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(2);
    matrix[0][0] = k;
    matrix[0][1] = -k;
    matrix[1][0] = -k;
    matrix[1][1] = k;
    matrix
}

/// Two subdomains sharing a corner (node 99, explicit, pins out the rigid-body
/// mode) and a dual interface node (node 50) — the minimal setup with a
/// nontrivial dual problem.
fn subdomains(stiffnesses: [f64; 2]) -> Vec<Subdomain<()>> {
    let partition = Partition::new(vec![vec![99, 50], vec![99, 50]]);
    let corners = CornerSelection::new(vec![99]);
    let (interfaces, _) = build_interfaces(&partition, &corners, 1);
    let splits = build_splits(&partition, &corners, 1);
    interfaces
        .into_iter()
        .zip(splits.iter())
        .zip(stiffnesses.iter())
        .map(|((interface, split), &k)| {
            let dual_dofs = split.dual().to_vec();
            let stiffness = stiffness(k);
            let k_dd: SquareMatrix = dual_dofs
                .iter()
                .map(|&row| dual_dofs.iter().map(|&col| stiffness[row][col]).collect())
                .collect();
            let dual_factor = k_dd.factorize_lu().unwrap();
            Subdomain::new((), interface, dual_factor, dual_dofs, 2)
        })
        .collect()
}

#[test]
fn dual_action_matches_the_hand_derived_operator() {
    let subdomains = subdomains([2.0, 3.0]);
    let lambda: Vector = [1.0].into_iter().collect();
    let f_lambda = dual_action(&subdomains, &lambda);
    // F = B_0 K_dd,0^-1 B_0^T + B_1 K_dd,1^-1 B_1^T = 1/2 + 1/3 = 5/6.
    assert!((f_lambda[0] - 5.0 / 6.0).abs() < 1e-12);
}

#[test]
fn projected_pcg_solves_the_dual_problem() {
    let subdomains = subdomains([2.0, 3.0]);
    let rhs: Vector = [1.0].into_iter().collect();
    let lambda = projected_pcg(&subdomains, &rhs).unwrap();
    // F * lambda = rhs, F = 5/6, so lambda = 6/5.
    assert!((lambda[0] - 1.2).abs() < 1e-10);
}
