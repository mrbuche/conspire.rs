use super::{Subdomain, dual_action, dual_operator, dual_precondition, projected_pcg};
use crate::domain::block::feti::{
    dual_primal::{
        CornerSelection, build_splits, coarse,
        condense::{Condensed, condense},
    },
    interface::{Partition, build_interfaces},
};
use crate::math::{SquareMatrix, Vector};

fn stiffness(entries: [[f64; 2]; 2]) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(2);
    (0..2).for_each(|i| (0..2).for_each(|j| matrix[i][j] = entries[i][j]));
    matrix
}

struct Setup {
    subdomains: Vec<Subdomain<()>>,
    schur: SquareMatrix,
}

/// Two subdomains sharing a corner (node 99, explicit, pins out the rigid-body
/// mode) and a dual interface node (node 50) — the minimal setup with a
/// nontrivial coarse (corner) AND dual problem. Stiffnesses are arbitrary SPD
/// 2x2, not a degenerate rank-1 bar — a rank-1 bar's corner Schur complement
/// vanishes identically, which would make the coarse problem singular.
fn setup() -> Setup {
    let partition = Partition::new(vec![vec![99, 50], vec![99, 50]]);
    let corners = CornerSelection::new(vec![99]);
    let (interfaces, _) = build_interfaces(&partition, &corners, 1);
    let splits = build_splits(&partition, &corners, 1);
    let stiffnesses = [
        stiffness([[4.0, 1.0], [1.0, 3.0]]),
        stiffness([[5.0, 2.0], [2.0, 4.0]]),
    ];
    let zero_force = Vector::zero(2);
    let condensed: Vec<Condensed> = splits
        .iter()
        .zip(stiffnesses.iter())
        .map(|(split, stiffness)| condense(stiffness, &zero_force, split))
        .collect();
    let (schur, _) = coarse::assemble(&condensed, &splits, &corners, 1);
    let subdomains = interfaces
        .into_iter()
        .zip(splits.iter())
        .zip(stiffnesses.iter())
        .zip(condensed.iter())
        .map(|(((interface, split), stiffness), condensed)| {
            let dual_dofs = split.dual().to_vec();
            let k_dd: SquareMatrix = dual_dofs
                .iter()
                .map(|&row| dual_dofs.iter().map(|&col| stiffness[row][col]).collect())
                .collect();
            let dual_factor = k_dd.factorize_lu().unwrap();
            Subdomain::new(
                (),
                interface,
                k_dd,
                dual_factor,
                dual_dofs,
                2,
                condensed.dual_map.clone(),
                split.primal_global().to_vec(),
            )
        })
        .collect();
    Setup { subdomains, schur }
}

#[test]
fn dual_action_matches_the_hand_derived_operator() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let f_lambda = dual_action(&setup.subdomains, &lambda);
    // F = 1/K_dd,0 + 1/K_dd,1 = 1/3 + 1/4 = 7/12.
    assert!((f_lambda[0] - 7.0 / 12.0).abs() < 1e-12);
}

#[test]
fn dual_operator_includes_the_coarse_coupling_correction() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let f_aug_lambda = dual_operator(&setup.subdomains, &lambda, &setup.schur);
    // Hand-derived: F = 7/12, S_pp = 23/3, C^T.1 = -1/6, C.(S_pp^-1.C^T) = 1/276.
    // F_aug = 7/12 + 1/276 = 27/46.
    assert!((f_aug_lambda[0] - 27.0 / 46.0).abs() < 1e-10);
}

#[test]
fn lumped_preconditioner_matches_the_hand_derived_operator() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let preconditioned = dual_precondition(&setup.subdomains, &lambda);
    // sum_s B_s K_dd,s B_s^T . 1 = K_dd,0 + K_dd,1 = 3 + 4 = 7.
    assert!((preconditioned[0] - 7.0).abs() < 1e-12);
}

#[test]
fn projected_pcg_solves_the_augmented_dual_problem() {
    let setup = setup();
    let rhs: Vector = [1.0].into_iter().collect();
    let lambda = projected_pcg(&setup.subdomains, &setup.schur, &rhs).unwrap();
    // F_aug * lambda = rhs, F_aug = 27/46, so lambda = 46/27.
    assert!((lambda[0] - 46.0 / 27.0).abs() < 1e-8);
}
