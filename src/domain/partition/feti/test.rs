use super::THREADS;
use crate::{
    domain::feti::{
        dual::{
            dual_action, dual_operator, dual_precondition, dual_precondition_dirichlet, dual_reduce,
        },
        dual_primal::{
            BoundaryConditions, CornerSelection, build_splits,
            coarse::{Coarse, CoarseSystem},
            condense::Condensed,
        },
        interface::build_interfaces,
        pcg::{primal_recovery, projected_pcg},
        subdomain::{DirichletLocal, Subdomain},
    },
    geometry::mesh::Partition,
    math::{SquareMatrix, Tensor, Vector},
};
use std::{
    collections::HashSet,
    sync::Mutex,
    thread::{available_parallelism, current, sleep},
    time::Duration,
};

fn stiffness(entries: [[f64; 2]; 2]) -> SquareMatrix {
    let mut matrix = SquareMatrix::zero(2);
    (0..2).for_each(|i| (0..2).for_each(|j| matrix[i][j] = entries[i][j]));
    matrix
}

fn condense(
    local_stiffness: &SquareMatrix,
    local_force: &Vector,
    primal: &[usize],
    dual: &[usize],
) -> Condensed {
    Condensed::try_condense(local_stiffness, local_force, primal, dual)
        .expect("remainder block K_dd is singular")
}

struct Setup {
    subdomains: Vec<Subdomain<()>>,
    coarse: Coarse,
    num_multipliers: usize,
}

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
    let (schur, _) = CoarseSystem::assemble(&condensed, &splits, &corner_dofs);
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
            let dirichlet = DirichletLocal::build(stiffness, &dual_dofs, interface.dofs());
            Subdomain::new(
                (),
                interface,
                k_dd,
                dual_factor,
                dual_dofs,
                2,
                condensed.dual_map.clone(),
                condensed.primal_map.clone(),
                split.primal().to_vec(),
                split.primal_global().to_vec(),
                dirichlet,
            )
        })
        .collect();
    Setup {
        subdomains,
        coarse: Coarse::new(schur),
        num_multipliers,
    }
}

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
    let (schur, _) = CoarseSystem::assemble(&condensed, &splits, &corner_dofs);
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
            let dirichlet = DirichletLocal::build(stiffness, &dual_dofs, interface.dofs());
            Subdomain::new(
                (),
                interface,
                k_dd,
                dual_factor,
                dual_dofs,
                2,
                condensed.dual_map.clone(),
                condensed.primal_map.clone(),
                split.primal().to_vec(),
                split.primal_global().to_vec(),
                dirichlet,
            )
        })
        .collect();
    Setup {
        subdomains,
        coarse: Coarse::new(schur),
        num_multipliers,
    }
}

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
    let count = 8;
    let setup = chain_setup(count);
    let lambda: Vector = (0..setup.num_multipliers).map(|i| 1.0 + i as f64).collect();
    let parallel = dual_action(&setup.subdomains, &lambda, 4);
    let serial = serial_dual_action(&setup.subdomains, &lambda, setup.num_multipliers);
    assert_eq!(parallel.len(), serial.len());
    parallel
        .iter()
        .zip(serial.iter())
        .for_each(|(&p, &s)| assert!((p - s).abs() < 1e-12));
}

#[test]
fn dual_reduce_uses_no_more_than_the_thread_cap() {
    let setup = chain_setup(16);
    let lambda: Vector = (0..setup.num_multipliers).map(|i| 1.0 + i as f64).collect();
    let threads = Mutex::new(HashSet::new());
    let max_threads = 4;
    dual_reduce(&setup.subdomains, &lambda, max_threads, |subdomain, rhs| {
        threads.lock().unwrap().insert(current().id());
        sleep(Duration::from_millis(2));
        subdomain.local_solve(rhs)
    });
    let used = threads.lock().unwrap().len();
    assert!(used <= max_threads, "used {used} threads");
    if available_parallelism().map_or(1, |n| n.get()) > 1 {
        assert!(used > 1, "never left the calling thread");
    }
}

#[test]
fn dual_action_matches_the_hand_derived_operator() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let f_lambda = dual_action(&setup.subdomains, &lambda, THREADS);
    assert!((f_lambda[0] - 7.0 / 12.0).abs() < 1e-12);
}

#[test]
fn dual_operator_includes_the_coarse_coupling_correction() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let f_aug_lambda = dual_operator(&setup.subdomains, &lambda, &setup.coarse, THREADS);
    assert!((f_aug_lambda[0] - 27.0 / 46.0).abs() < 1e-10);
}

#[test]
fn lumped_preconditioner_matches_the_hand_derived_operator() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let preconditioned = dual_precondition(&setup.subdomains, &lambda, THREADS);
    assert!((preconditioned[0] - 7.0).abs() < 1e-12);
}

#[test]
fn dirichlet_preconditioner_matches_lumped_when_every_dual_dof_is_on_the_interface() {
    let setup = setup();
    let lambda: Vector = [1.0].into_iter().collect();
    let lumped = dual_precondition(&setup.subdomains, &lambda, THREADS);
    let dirichlet = dual_precondition_dirichlet(&setup.subdomains, &lambda, THREADS);
    assert!((dirichlet[0] - lumped[0]).abs() < 1e-12);
}

#[test]
fn projected_pcg_solves_the_augmented_dual_problem() {
    let setup = setup();
    let rhs: Vector = [1.0].into_iter().collect();
    let lambda = projected_pcg(&setup.subdomains, &setup.coarse, &rhs).unwrap();
    assert!((lambda[0] - 46.0 / 27.0).abs() < 1e-8);
}

#[test]
fn primal_recovery_matches_the_hand_derived_solution() {
    let setup = setup();
    let local_forces = [Vector::zero(2), Vector::zero(2)];
    let corner_solution: Vector = [-1.0 / 46.0].into_iter().collect();
    let lambda: Vector = [1.0].into_iter().collect();
    let recovered = primal_recovery(&setup.subdomains, &local_forces, &corner_solution, &lambda);
    assert!((recovered[0][0] - (-1.0 / 46.0)).abs() < 1e-10);
    assert!((recovered[0][1] - (-15.0 / 46.0)).abs() < 1e-10);
    assert!((recovered[1][0] - (-1.0 / 46.0)).abs() < 1e-10);
    assert!((recovered[1][1] - (6.0 / 23.0)).abs() < 1e-10);
}
