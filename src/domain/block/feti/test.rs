use super::{
    Subdomain, dual_action, dual_operator, dual_precondition, primal_recovery, projected_pcg,
};
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
                split.primal().to_vec(),
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

#[test]
fn primal_recovery_matches_the_hand_derived_solution() {
    let setup = setup();
    // Zero forces (as in setup()), lambda = 1: corner_solution = S_pp^-1 . C^T.1
    // = -1/46, from the same derivation as dual_operator's F_aug test.
    let local_forces = [Vector::zero(2), Vector::zero(2)];
    let corner_solution: Vector = [-1.0 / 46.0].into_iter().collect();
    let lambda: Vector = [1.0].into_iter().collect();
    let recovered = primal_recovery(&setup.subdomains, &local_forces, &corner_solution, &lambda);
    // Hand-derived (verified by substitution back into both subdomains' local
    // equilibrium and the assembled corner equilibrium):
    // u0 = [-1/46, -15/46], u1 = [-1/46, 6/23].
    assert!((recovered[0][0] - (-1.0 / 46.0)).abs() < 1e-10);
    assert!((recovered[0][1] - (-15.0 / 46.0)).abs() < 1e-10);
    assert!((recovered[1][0] - (-1.0 / 46.0)).abs() < 1e-10);
    assert!((recovered[1][1] - (6.0 / 23.0)).abs() < 1e-10);
}

#[cfg(feature = "fem")]
mod solve_test {
    use super::super::solve;
    use crate::{
        constitutive::solid::elastic::{
            AlmansiHamelEulerian,
            test::{BULK_MODULUS, SHEAR_MODULUS},
        },
        domain::block::feti::interface::Partition,
        fem::{
            NodalCoordinates, NodalReferenceCoordinates,
            block::{Block, element::linear::Tetrahedron},
        },
    };

    /// Three tetrahedra sharing one face {1,2,3} — each subdomain is one
    /// element, and since all three meet at nodes 1, 2 and 3, the standard
    /// heuristic (shared by >= 3 subdomains) makes all three corners, pinning
    /// out each SUBDOMAIN's local rigid-body modes. Coordinates: nodes 0-3
    /// are the codebase's standard reference tetrahedron; nodes 4 and 5 are
    /// chosen so elements [1,2,3,4] and [1,2,3,5] both have positive volume.
    ///
    /// This assembly is completely free-floating — no Dirichlet boundary
    /// condition pins it anywhere in space — so it still has 6 GLOBAL
    /// rigid-body modes. Corner condensation only removes a subdomain's own
    /// local floating modes; it can't remove a mode that moves the whole
    /// structure together, since a Schur complement can't have lower rank
    /// than the directions of the original system's null space that survive
    /// projection onto the corner DOFs. `solve()` has no boundary-condition
    /// mechanism yet, so the assembled coarse problem is correctly singular
    /// here, and `solve()` correctly refuses rather than returning garbage.
    fn coordinates() -> Vec<[f64; 3]> {
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 2.0],
        ]
    }

    fn block() -> Block<AlmansiHamelEulerian, Tetrahedron, 1, 3, 4, 4> {
        let reference_coordinates = NodalReferenceCoordinates::from(coordinates());
        Block::from((
            AlmansiHamelEulerian {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            },
            vec![[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 5]],
            &reference_coordinates,
        ))
    }

    /// A free-floating assembly with no Dirichlet boundary condition has
    /// global rigid-body modes that corner condensation alone can't remove
    /// (see `coordinates`' doc comment) — `solve()` correctly refuses this
    /// rather than silently returning garbage. Runs the whole pipeline
    /// (real-Block extraction through condensation, coarse assembly, dual
    /// PCG wiring, and primal recovery) up to that point without panicking
    /// anywhere else, which is what this test actually exercises; a
    /// well-posed (externally supported) end-to-end solve needs boundary
    /// condition support that doesn't exist yet.
    #[test]
    #[should_panic(expected = "singular")]
    fn a_free_floating_assembly_has_no_boundary_condition_to_pin_it() {
        let block = block();
        let nodal_coordinates = NodalCoordinates::from(coordinates());
        let partition = Partition::new(vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![1, 2, 3, 5]]);
        let _ = solve(&block, &nodal_coordinates, &partition, 3);
    }
}
