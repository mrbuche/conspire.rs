use super::{
    SETUP_THREADS, Subdomain, dirichlet_local, dual_action, dual_operator, dual_precondition,
    dual_precondition_dirichlet, parallel_map, primal_recovery, projected_pcg,
};
use crate::domain::block::feti::{
    dual_primal::{
        BoundaryConditions, CornerSelection, build_splits,
        coarse::{self, Coarse},
        condense::{Condensed, condense},
    },
    interface::build_interfaces,
};
use crate::geometry::mesh::Partition;
use crate::math::{SquareMatrix, Tensor, Vector};

fn stiffness(entries: [[f64; 2]; 2]) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(2);
    (0..2).for_each(|i| (0..2).for_each(|j| matrix[i][j] = entries[i][j]));
    matrix
}

struct Setup {
    subdomains: Vec<Subdomain<()>>,
    coarse: Coarse,
    num_multipliers: usize,
}

/// Two subdomains sharing a corner (node 99, explicit, pins out the rigid-body
/// mode) and a dual interface node (node 50) — the minimal setup with a
/// nontrivial coarse (corner) AND dual problem. Stiffnesses are arbitrary SPD
/// 2x2, not a degenerate rank-1 bar — a rank-1 bar's corner Schur complement
/// vanishes identically, which would make the coarse problem singular.
fn setup() -> Setup {
    let partition = Partition::from_parts_nodes(vec![vec![99, 50], vec![99, 50]]);
    let corners = CornerSelection::new(vec![99]);
    let (interfaces, num_multipliers) = build_interfaces(&partition, &corners, 1);
    let (splits, corner_dofs) = build_splits(&partition, &corners, &BoundaryConditions::none(), 1);
    let stiffnesses = [
        stiffness([[4.0, 1.0], [1.0, 3.0]]),
        stiffness([[5.0, 2.0], [2.0, 4.0]]),
    ];
    let zero_force = Vector::zero(2);
    let condensed: Vec<Condensed> = splits
        .iter()
        .zip(stiffnesses.iter())
        .map(|(split, stiffness)| condense(stiffness, &zero_force, split.primal(), split.dual()))
        .collect();
    let (schur, _) = coarse::assemble(&condensed, &splits, &corner_dofs);
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
            let (boundary_dofs, dirichlet_schur) =
                dirichlet_local(stiffness, &dual_dofs, interface.dofs());
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
                boundary_dofs,
                dirichlet_schur,
            )
        })
        .collect();
    Setup {
        subdomains,
        coarse: Coarse::new(&schur),
        num_multipliers,
    }
}

/// A chain of `count` subdomains, nodes `0..=count`, subdomain `i`
/// connecting node `i` to node `i+1`. Corners are the even-indexed nodes,
/// dual (interface) nodes the odd-indexed ones — consecutive integers
/// always have opposite parity, so every subdomain gets exactly one corner
/// and one dual node regardless of `count`, the same well-posed shape as
/// `setup()`'s 2-subdomain example, just repeated. Distinct, non-rank-1 SPD
/// stiffness per subdomain (as in `setup()`, avoiding the degenerate
/// corner-Schur-vanishes case a plain bar would hit).
fn chain_setup(count: usize) -> Setup {
    let partition = Partition::from_parts_nodes((0..count).map(|i| vec![i, i + 1]).collect());
    let corners = CornerSelection::new((0..=count).step_by(2).collect());
    let (interfaces, num_multipliers) = build_interfaces(&partition, &corners, 1);
    let (splits, corner_dofs) = build_splits(&partition, &corners, &BoundaryConditions::none(), 1);
    let stiffnesses: Vec<SquareMatrix> = (0..count)
        .map(|i| stiffness([[i as f64 + 4.0, 1.0], [1.0, i as f64 + 3.0]]))
        .collect();
    let zero_force = Vector::zero(2);
    let condensed: Vec<Condensed> = splits
        .iter()
        .zip(stiffnesses.iter())
        .map(|(split, stiffness)| condense(stiffness, &zero_force, split.primal(), split.dual()))
        .collect();
    let (schur, _) = coarse::assemble(&condensed, &splits, &corner_dofs);
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
            let (boundary_dofs, dirichlet_schur) =
                dirichlet_local(stiffness, &dual_dofs, interface.dofs());
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
                boundary_dofs,
                dirichlet_schur,
            )
        })
        .collect();
    Setup {
        subdomains,
        coarse: Coarse::new(&schur),
        num_multipliers,
    }
}

/// Reference (deliberately serial, no threading) recomputation of
/// `dual_reduce`'s reduction — bypasses `dual_action` entirely by calling
/// the same private `Subdomain` primitives directly, so this is an
/// independent check on the parallel path, not a re-test of the same code.
fn serial_dual_action(
    subdomains: &[Subdomain<()>],
    lambda: &Vector,
    num_multipliers: usize,
) -> Vector {
    subdomains
        .iter()
        .map(|subdomain| {
            let rhs = subdomain.interface().apply_transpose(lambda, 2);
            let local = subdomain.local_solve(&rhs);
            subdomain.interface().apply(&local, num_multipliers)
        })
        .fold(Vector::zero(num_multipliers), |sum, contribution| {
            sum + contribution
        })
}

#[test]
fn dual_reduce_parallel_path_matches_serial_reference() {
    // 8 >= PARALLEL_THRESHOLD, so dual_action here genuinely exercises the
    // multithreaded path in dual_reduce, not just the serial fallback every
    // other test in this file uses (all well below the threshold).
    let count = 8;
    let setup = chain_setup(count);
    let lambda: Vector = (0..setup.num_multipliers).map(|i| 1.0 + i as f64).collect();
    let parallel = dual_action(&setup.subdomains, &lambda);
    let serial = serial_dual_action(&setup.subdomains, &lambda, setup.num_multipliers);
    assert_eq!(parallel.len(), serial.len());
    parallel
        .iter()
        .zip(serial.iter())
        .for_each(|(&p, &s)| assert!((p - s).abs() < 1e-12));
}

#[test]
fn parallel_map_keeps_item_order_under_uneven_work() {
    let items: Vec<usize> = (0..53).collect();
    let mapped = parallel_map(&items, |&item| {
        if item % 7 == 0 {
            std::thread::sleep(std::time::Duration::from_millis(3));
        }
        item * item
    });
    assert_eq!(mapped, items.iter().map(|&i| i * i).collect::<Vec<_>>());
}

#[test]
fn parallel_map_uses_no_more_than_the_setup_thread_cap() {
    let items: Vec<usize> = (0..64).collect();
    let threads = std::sync::Mutex::new(std::collections::HashSet::new());
    parallel_map(&items, |_| {
        threads.lock().unwrap().insert(std::thread::current().id());
        std::thread::sleep(std::time::Duration::from_millis(2));
    });
    let used = threads.lock().unwrap().len();
    assert!(used <= SETUP_THREADS, "used {used} threads");
    if std::thread::available_parallelism().map_or(1, |n| n.get()) > 1 {
        assert!(used > 1, "never left the calling thread");
    }
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
    let f_aug_lambda = dual_operator(&setup.subdomains, &lambda, &setup.coarse);
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

/// Independent hand-derived check of `dirichlet_local`'s classification and
/// Schur complement, on a 3-dof local stiffness reused from
/// `condense`'s own hand-derived test — dual dofs {1, 2}, but only dof 1 is
/// declared on the interface here, so dof 2 is interior and gets eliminated:
/// S_GammaGamma = K_11 - K_12 . K_22^-1 . K_21 = 4 - 1*(1/4)*1 = 3.75.
#[test]
fn dirichlet_local_splits_interior_and_boundary_and_computes_the_schur_complement() {
    let stiffness: SquareMatrix = [[4.0, 1.0, 0.0], [1.0, 4.0, 1.0], [0.0, 1.0, 4.0]]
        .into_iter()
        .map(|row| row.into_iter().collect())
        .collect();
    let dual_dofs = vec![1, 2];
    let interface_dofs = vec![1];
    let (boundary_dofs, schur) = dirichlet_local(&stiffness, &dual_dofs, &interface_dofs);
    assert_eq!(boundary_dofs, vec![1]);
    assert!((schur[0][0] - 3.75).abs() < 1e-12);
}

/// `setup()`'s subdomains each have exactly one dual dof, and it's always on
/// the interface (no interior dof to eliminate) — so `S_GammaGamma = K_dd`
/// exactly and the Dirichlet preconditioner must coincide with the lumped
/// one here, even though the two are generally different reductions.
#[test]
fn dirichlet_preconditioner_matches_lumped_when_every_dual_dof_is_on_the_interface() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let lumped = dual_precondition(&setup.subdomains, &lambda);
    let dirichlet = dual_precondition_dirichlet(&setup.subdomains, &lambda);
    assert!((dirichlet[0] - lumped[0]).abs() < 1e-12);
}

#[test]
fn projected_pcg_solves_the_augmented_dual_problem() {
    let setup = setup();
    let rhs: Vector = [1.0].into_iter().collect();
    let lambda = projected_pcg(&setup.subdomains, &setup.coarse, &rhs).unwrap();
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
        constitutive::solid::{
            elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
            hyperelastic::NeoHookean,
        },
        domain::block::feti::dual_primal::BoundaryConditions,
        fem::{
            NodalCoordinates, NodalReferenceCoordinates,
            block::{Block, element::linear::Tetrahedron},
        },
        geometry::mesh::Partition,
        math::Tensor,
    };

    /// Three tetrahedra sharing one face {1,2,3} — each subdomain is one
    /// element, and since all three meet at nodes 1, 2 and 3, the standard
    /// heuristic (shared by >= 3 subdomains) makes all three corners, pinning
    /// out each SUBDOMAIN's local rigid-body modes. Coordinates: nodes 0-3
    /// are the codebase's standard reference tetrahedron; nodes 4 and 5 are
    /// chosen so elements [1,2,3,4] and [1,2,3,5] both have positive volume.
    ///
    /// With no boundary conditions applied, this assembly is completely
    /// free-floating in space and so still has 6 GLOBAL rigid-body modes.
    /// Corner condensation only removes a subdomain's own local floating
    /// modes; it can't remove a mode that moves the whole structure
    /// together, since a Schur complement can't have lower rank than the
    /// directions of the original system's null space that survive
    /// projection onto the corner DOFs — see the two tests below for both
    /// the unsupported (singular, correctly refused) and supported
    /// (well-posed) cases.
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

    fn block() -> Block<NeoHookean, Tetrahedron, 1, 3, 4, 4> {
        let reference_coordinates = NodalReferenceCoordinates::from(coordinates());
        Block::from((
            NeoHookean {
                bulk_modulus: BULK_MODULUS,
                shear_modulus: SHEAR_MODULUS,
            },
            vec![[0, 1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 5]],
            &reference_coordinates,
        ))
    }

    fn partition() -> Partition {
        Partition::from_parts_nodes(vec![vec![0, 1, 2, 3], vec![1, 2, 3, 4], vec![1, 2, 3, 5]])
    }

    /// A free-floating assembly with no Dirichlet boundary condition has
    /// global rigid-body modes that corner condensation alone can't remove
    /// (see `coordinates`' doc comment) — `solve()` correctly refuses this
    /// rather than silently returning garbage. Runs the whole pipeline
    /// (real-Block extraction through condensation, coarse assembly, dual
    /// PCG wiring, and primal recovery) up to that point without panicking
    /// anywhere else, which is what this test actually exercises.
    #[test]
    #[should_panic(expected = "singular")]
    fn a_free_floating_assembly_has_no_boundary_condition_to_pin_it() {
        let block = block();
        let nodal_coordinates = NodalCoordinates::from(coordinates());
        let _ = solve(
            &block,
            &nodal_coordinates,
            &partition(),
            &BoundaryConditions::none(),
            3,
        );
    }

    /// Fixes nodes 0, 4 and 5 — the three exclusive apex nodes, one per
    /// subdomain, none of them shared/corner nodes — completely. Three
    /// non-collinear fully-fixed points remove all 6 global rigid-body
    /// modes (more than sufficient; redundant on the translations).
    ///
    /// With zero applied force and every prescribed value equal to the
    /// reference position, the unique equilibrium is zero displacement
    /// everywhere.
    #[test]
    fn a_supported_assembly_at_zero_deformation_solves_to_zero_displacement() {
        let block = block();
        let nodal_coordinates = NodalCoordinates::from(coordinates());
        let boundary_conditions = BoundaryConditions::new(vec![
            (0, 0),
            (0, 1),
            (0, 2),
            (4, 0),
            (4, 1),
            (4, 2),
            (5, 0),
            (5, 1),
            (5, 2),
        ]);
        let solution = solve(
            &block,
            &nodal_coordinates,
            &partition(),
            &boundary_conditions,
            3,
        )
        .unwrap_or_else(|_| panic!("solve failed"));
        assert_eq!(solution.len(), 6 * 3);
        solution.iter().for_each(|&entry| {
            assert!(entry.is_finite());
            assert!(entry.abs() < 1e-8);
        });
    }

    /// Fixes node 0 (exclusive) fully, plus node 1 and node 2 — both
    /// CORNER nodes, shared by all three subdomains — fully and z-only
    /// respectively: node0(xyz) removes translations, node1(xyz) removes 2
    /// rotations (about axes through both node0 and node1), node2(z)
    /// removes the third. This is the same scheme an earlier version of
    /// this test used when `coarse::assemble` still sized the coarse
    /// problem from raw corner-node count rather than the DOFs that
    /// actually survive boundary conditions — it hit a singular coarse
    /// problem then purely from that sizing bug, now fixed
    /// (`CornerDofs`), not from anything wrong with this boundary
    /// condition placement. This test is the direct regression check for
    /// that fix: boundary conditions on corner-node components now work.
    #[test]
    fn a_boundary_condition_on_a_corner_node_solves_correctly() {
        let block = block();
        let nodal_coordinates = NodalCoordinates::from(coordinates());
        let boundary_conditions =
            BoundaryConditions::new(vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 2)]);
        let solution = solve(
            &block,
            &nodal_coordinates,
            &partition(),
            &boundary_conditions,
            3,
        )
        .unwrap_or_else(|_| panic!("solve failed"));
        assert_eq!(solution.len(), 6 * 3);
        solution.iter().for_each(|&entry| {
            assert!(entry.is_finite());
            assert!(entry.abs() < 1e-8);
        });
    }
}
