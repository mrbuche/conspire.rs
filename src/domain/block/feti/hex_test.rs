use super::{
    Preconditioner, SolveStats,
    assemble::local_stiffness_and_force,
    dirichlet_local,
    dual_primal::{CornerSelection, build_splits, condense::condense},
    interface::build_interfaces,
    solve_with,
};
use crate::{
    constitutive::solid::{
        elastic::test::{BULK_MODULUS, SHEAR_MODULUS},
        hyperelastic::NeoHookean,
    },
    domain::{
        block::{
            element::Elements, feti::dual_primal::BoundaryConditions, finalize_node_neighbors,
            solver_from_neighbors,
        },
        nodal_coordinates,
        solid::{elastic::ElasticElements, hyperelastic::HyperelasticElements},
    },
    fem::{
        NodalCoordinates,
        block::{Block, element::linear::Hexahedron},
    },
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh, Partition},
    },
    math::{
        SquareMatrix, Tensor, Vector,
        optimize::{EqualityConstraint, NewtonRaphson, SecondOrderOptimization},
    },
};
use std::{
    cell::Cell,
    time::{Duration, Instant},
};

type HexBlock = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;

struct Fixture {
    block: HexBlock,
    nodal_coordinates: NodalCoordinates<3>,
    partition: Partition,
    boundary_conditions: BoundaryConditions,
    fixed: Vec<(usize, usize)>,
    num_nodes: usize,
}

fn perturbed(reference: [f64; 3], extent: f64) -> [f64; 3] {
    let [x, y, z] = reference;
    [
        x + 0.02 * y + 0.005 * y * z / extent,
        y + 0.01 * z,
        z + 0.015 * x * x / extent,
    ]
}

/// A voxel hex mesh of `nel` elements split by `partition_box(divisions)`,
/// with the `x = 0` face fully fixed and the free nodes moved to a smooth
/// non-affine configuration, so that the Newton step has a nonzero
/// right-hand side everywhere.
fn fixture(nel: [usize; 3], divisions: [usize; 3]) -> Fixture {
    let mesh = Mesh::from_voxels(Voxels::new(vec![1u8; nel.iter().product()], nel), None);
    let partition = mesh.partition_box(divisions);
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh.into();
    let reference: Vec<[f64; 3]> = coordinates
        .iter()
        .map(|c| [c[0].value(), c[1].value(), c[2].value()])
        .collect();
    let elements: Vec<[usize; 8]> = connectivities
        .into_members()
        .into_iter()
        .flat_map(|connectivity| match connectivity {
            Connectivity::Hexahedral(hexahedra) => hexahedra.into_iter().collect::<Vec<_>>(),
            _ => panic!("expected a hexahedral mesh"),
        })
        .collect();
    let extent = nel[0] as f64;
    let block = HexBlock::from((
        NeoHookean {
            bulk_modulus: BULK_MODULUS,
            shear_modulus: SHEAR_MODULUS,
        },
        elements,
        &nodal_coordinates(coordinates),
    ));
    let current: Vec<[f64; 3]> = reference
        .iter()
        .map(|&point| perturbed(point, extent))
        .collect();
    let fixed: Vec<(usize, usize)> = reference
        .iter()
        .enumerate()
        .filter(|(_, point)| point[0].abs() < 1e-9)
        .flat_map(|(node, _)| (0..3).map(move |component| (node, component)))
        .collect();
    Fixture {
        block,
        nodal_coordinates: NodalCoordinates::from(current),
        partition,
        boundary_conditions: BoundaryConditions::new(fixed.clone()),
        fixed,
        num_nodes: reference.len(),
    }
}

/// The whole mesh assembled as ONE subdomain, so the reference solve
/// shares only the element routines with the FETI path.
fn global_system(fixture: &Fixture) -> (SquareMatrix, Vector) {
    let all: Vec<usize> = (0..fixture.num_nodes).collect();
    local_stiffness_and_force(&fixture.block, &fixture.nodal_coordinates, &all)
        .unwrap_or_else(|_| panic!("global assembly failed"))
}

/// Largest `|K_ij - K_ji|` relative to the largest `|K_ij|`.
fn tangent_asymmetry(stiffness: &SquareMatrix) -> f64 {
    let n = stiffness.len();
    let mut largest = 0.0_f64;
    let mut skew = 0.0_f64;
    (0..n).for_each(|i| {
        (0..n).for_each(|j| {
            largest = largest.max(stiffness[i][j].abs());
            skew = skew.max((stiffness[i][j] - stiffness[j][i]).abs());
        })
    });
    skew / largest
}

/// Dense LU solve of `K_ff u_f = f_f` over the free dofs.
fn dense_oracle(fixture: &Fixture, stiffness: &SquareMatrix, force: &Vector) -> Vec<f64> {
    let is_fixed = |dof: usize| fixture.fixed.contains(&(dof / 3, dof % 3));
    let free: Vec<usize> = (0..3 * fixture.num_nodes)
        .filter(|&dof| !is_fixed(dof))
        .collect();
    let k_ff: SquareMatrix = free
        .iter()
        .map(|&row| free.iter().map(|&col| stiffness[row][col]).collect())
        .collect();
    let f_f: Vector = free.iter().map(|&dof| force[dof]).collect();
    let u_f = k_ff.solve_lu(&f_f).expect("global stiffness is singular");
    let mut solution = vec![0.0; 3 * fixture.num_nodes];
    free.iter()
        .zip(u_f.iter())
        .for_each(|(&dof, &value)| solution[dof] = value);
    solution
}

fn run(fixture: &Fixture, preconditioner: Preconditioner) -> (Vec<f64>, SolveStats) {
    let (solution, stats) = solve_with(
        &fixture.block,
        &fixture.nodal_coordinates,
        &fixture.partition,
        &fixture.boundary_conditions,
        3,
        preconditioner,
        1e-10,
    )
    .unwrap_or_else(|_| panic!("solve failed"));
    (solution.iter().copied().collect(), stats)
}

fn relative_error(reference: &[f64], solution: &[f64]) -> f64 {
    let scale = reference.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    assert!(scale > 1e-8, "oracle solution is trivially zero");
    reference
        .iter()
        .zip(solution.iter())
        .fold(0.0_f64, |m, (&r, &s)| m.max((r - s).abs()))
        / scale
}

/// Every decomposed solve must reproduce the dense global solve; returns the
/// dual operator applications each preconditioner took.
fn check(nel: [usize; 3], divisions: [usize; 3]) -> (usize, usize) {
    let fixture = fixture(nel, divisions);
    let (stiffness, force) = global_system(&fixture);
    assert!(
        tangent_asymmetry(&stiffness) < 1e-12,
        "FETI-DP needs a symmetric tangent"
    );
    let reference = dense_oracle(&fixture, &stiffness, &force);
    let counts: Vec<usize> = [Preconditioner::Lumped, Preconditioner::Dirichlet]
        .into_iter()
        .map(|preconditioner| {
            let (solution, stats) = run(&fixture, preconditioner);
            let error = relative_error(&reference, &solution);
            println!(
                "{nel:?} in {divisions:?} parts, {preconditioner:?}: {} applications, \
                 relative error {error:e}",
                stats.applications
            );
            assert!(error < 1e-8, "relative error {error:e}");
            stats.applications
        })
        .collect();
    (counts[0], counts[1])
}

#[test]
fn a_decomposed_hex_mesh_matches_the_dense_global_solve() {
    check([6; 3], [2; 3]);
}

fn milliseconds(duration: std::time::Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

/// Runs both preconditioners on one decomposition and prints a row per
/// preconditioner. There is no dense oracle at this size, so correctness
/// rests on `check` above plus the two preconditioners agreeing.
fn benchmark(nel: [usize; 3], divisions: [usize; 3]) {
    let fixture = fixture(nel, divisions);
    let per_edge = nel[0] / divisions[0] + 1;
    let subdomain_dofs = 3 * per_edge.pow(3);
    let solutions: Vec<(Preconditioner, Vec<f64>, SolveStats)> =
        [Preconditioner::Lumped, Preconditioner::Dirichlet]
            .into_iter()
            .map(|preconditioner| {
                let (solution, stats) = run(&fixture, preconditioner);
                (preconditioner, solution, stats)
            })
            .collect();
    solutions.iter().for_each(|(preconditioner, _, stats)| {
        let total = stats.assemble
            + stats.condense
            + stats.subdomains
            + stats.coarse
            + stats.pcg
            + stats.recover;
        println!(
            "{:>10?} {:>9} {:>9} {:>9} {:>6} | asm {:>7.0} cond {:>7.0} subs {:>8.0} coarse {:>7.0} \
             pcg {:>8.0} rec {:>6.0} | total {:>8.0} ms",
            preconditioner,
            format!("{nel:?}").replace(' ', ""),
            format!("{divisions:?}").replace(' ', ""),
            subdomain_dofs,
            stats.applications,
            milliseconds(stats.assemble),
            milliseconds(stats.condense),
            milliseconds(stats.subdomains),
            milliseconds(stats.coarse),
            milliseconds(stats.pcg),
            milliseconds(stats.recover),
            milliseconds(total),
        )
    });
    let scale = solutions[0].1.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let disagreement = relative_error(&solutions[0].1, &solutions[1].1);
    assert!(scale > 1e-8, "solution is trivially zero");
    assert!(
        disagreement < 1e-6,
        "lumped and Dirichlet disagree by {disagreement:e}"
    );
}

const HEADER: &str = "preconditioner  mesh       parts   sub dofs  apps";

/// Fixed global mesh, more and smaller subdomains.
#[test]
#[ignore]
fn benchmark_subdomain_count() {
    println!("{HEADER}");
    [[3; 3], [4; 3], [6; 3]]
        .into_iter()
        .for_each(|divisions| benchmark([24; 3], divisions));
}

/// Fixed 3x3x3 subdomain grid, more elements per subdomain (H/h grows).
#[test]
#[ignore]
fn benchmark_subdomain_size() {
    println!("{HEADER}");
    [9, 12, 18, 24]
        .into_iter()
        .for_each(|nel| benchmark([nel; 3], [3; 3]));
}

/// The existing global path: `NewtonRaphson` minimizing the strain energy on
/// the sparse symmetric solver (what `Model::minimize` does for a
/// hyperelastic model), from the same perturbed configuration with the same
/// fixed face. Newton runs to convergence, so this reports the step count and
/// the assembly time alongside the total.
fn monolithic_baseline(nel: [usize; 3]) {
    let fixture = fixture(nel, [1; 3]);
    let coordinates = &fixture.nodal_coordinates;
    let constraint = EqualityConstraint::Fixed(
        fixture
            .fixed
            .iter()
            .map(|&(node, component)| 3 * node + component)
            .collect(),
    );
    let mut neighbors = vec![Vec::new(); coordinates.len()];
    fixture.block.node_neighbors(&mut neighbors);
    finalize_node_neighbors(&mut neighbors);
    let sparse = solver_from_neighbors(&neighbors, &constraint, 3, true);
    let steps = Cell::new(0_usize);
    let assembly = Cell::new(Duration::ZERO);
    let clock = Instant::now();
    NewtonRaphson::default()
        .minimize(
            |x: &NodalCoordinates<3>| Ok(fixture.block.helmholtz_free_energy(x)?),
            |x: &NodalCoordinates<3>| {
                let start = Instant::now();
                let forces = fixture.block.nodal_forces(x)?;
                assembly.set(assembly.get() + start.elapsed());
                Ok(forces)
            },
            |x: &NodalCoordinates<3>| {
                let start = Instant::now();
                let stiffnesses = fixture.block.nodal_stiffnesses_symmetric(x)?;
                assembly.set(assembly.get() + start.elapsed());
                steps.set(steps.get() + 1);
                Ok(stiffnesses)
            },
            coordinates.clone(),
            constraint,
            Some(sparse),
        )
        .unwrap_or_else(|_| panic!("monolithic solve failed"));
    let total = clock.elapsed();
    let steps = steps.get().max(1);
    println!(
        "{:>9} {:>8} dofs | {steps} Newton steps | total {:>8.0} ms, per step {:>8.0} ms, \
         element assembly {:>7.0} ms",
        format!("{nel:?}").replace(' ', ""),
        3 * (nel[0] + 1) * (nel[1] + 1) * (nel[2] + 1),
        milliseconds(total),
        milliseconds(total) / steps as f64,
        milliseconds(assembly.get()),
    );
}

#[test]
#[ignore]
fn benchmark_monolithic_baseline() {
    [9, 12, 18, 24, 30]
        .into_iter()
        .for_each(|nel| monolithic_baseline([nel; 3]));
}

/// 30^3 (89.4k dofs), the largest size the sparse baseline was run at. Local
/// matrices are dense, so memory grows with the number of subdomains: this
/// case holds about 125 subdomains of 1029 dofs (roughly 3.6 GB).
#[test]
#[ignore]
fn benchmark_scaling_30_large_subdomains() {
    println!("{HEADER}");
    benchmark([30; 3], [5; 3]);
}

/// The same mesh in 216 smaller subdomains of 648 dofs (roughly 2.5 GB).
#[test]
#[ignore]
fn benchmark_scaling_30_small_subdomains() {
    println!("{HEADER}");
    benchmark([30; 3], [6; 3]);
}

/// Times each step of one subdomain's setup, serially, on the subdomain with
/// the most dual dofs. The parallel stages can only be timed as a whole, so
/// this is where the split inside the subdomain build shows up. The
/// implicit Dirichlet application is also checked against the explicitly
/// formed Schur complement, which `condense` computes.
fn profile_subdomain_setup(nel: [usize; 3], divisions: [usize; 3]) {
    let fixture = fixture(nel, divisions);
    let partition = &fixture.partition;
    let corners = CornerSelection::from_partition(partition);
    let (interfaces, _) = build_interfaces(partition, &corners, 3);
    let (splits, _) = build_splits(partition, &corners, &fixture.boundary_conditions, 3);
    let index = (0..splits.len())
        .max_by_key(|&index| (splits[index].dual().len(), usize::MAX - index))
        .unwrap();
    let nodes = &partition.parts_nodes()[index];
    let (stiffness, force) =
        local_stiffness_and_force(&fixture.block, &fixture.nodal_coordinates, nodes)
            .unwrap_or_else(|_| panic!("assembly failed"));
    let (primal, dual) = (splits[index].primal(), splits[index].dual());
    let time = |work: &mut dyn FnMut()| {
        let start = Instant::now();
        work();
        milliseconds(start.elapsed())
    };
    let mut dual_stiffness = SquareMatrix::zero(0);
    let extract = time(&mut || {
        dual_stiffness = dual
            .iter()
            .map(|&row| dual.iter().map(|&col| stiffness[row][col]).collect())
            .collect();
    });
    let lu_dual = time(&mut || {
        dual_stiffness.factorize_lu().unwrap();
    });
    let corner_condense = time(&mut || {
        condense(&stiffness, &force, primal, dual);
    });
    let mut local = None;
    let dirichlet_setup = time(&mut || {
        local = Some(dirichlet_local(&stiffness, dual, interfaces[index].dofs()));
    });
    let local = local.unwrap();
    let boundary_dofs = local.boundary_dofs().to_vec();
    let interior: Vec<usize> = dual
        .iter()
        .copied()
        .filter(|dof| !boundary_dofs.contains(dof))
        .collect();
    let x: Vector = (0..boundary_dofs.len()).map(|i| 1.0 + i as f64).collect();
    let mut applied = Vector::zero(0);
    let apply = time(&mut || applied = local.apply(&x));
    let mut explicit = None;
    let explicit_schur = time(&mut || {
        explicit = Some(condense(&stiffness, &force, &boundary_dofs, &interior).schur);
    });
    let explicit = explicit.unwrap();
    let reference: Vec<f64> = (0..boundary_dofs.len())
        .map(|row| {
            (0..boundary_dofs.len())
                .map(|column| explicit[row][column] * x[column])
                .sum()
        })
        .collect();
    let disagreement = reference
        .iter()
        .zip(applied.iter())
        .fold(0.0_f64, |m, (&r, &a)| m.max((r - a).abs()))
        / reference.iter().fold(0.0_f64, |m, &r| m.max(r.abs()));
    assert!(
        disagreement < 1e-9,
        "implicit and explicit disagree by {disagreement:e}"
    );
    println!(
        "{:>9} {:>9} | dofs {:>5} dual {:>5} boundary {:>5} interior {:>5} corner {:>4} | \
         K_dd extract {:>6.0} LU {:>6.0} | corner condense {:>6.0} | Dirichlet: implicit setup \
         {:>6.0} + {:>6.1} per apply, vs explicit Schur {:>7.0} ms (agree to {disagreement:.0e})",
        format!("{nel:?}").replace(' ', ""),
        format!("{divisions:?}").replace(' ', ""),
        3 * nodes.len(),
        dual.len(),
        boundary_dofs.len(),
        interior.len(),
        primal.len(),
        extract,
        lu_dual,
        corner_condense,
        dirichlet_setup,
        apply,
        explicit_schur,
    );
}

#[test]
#[ignore]
fn profile_one_subdomain_setup() {
    profile_subdomain_setup([24; 3], [3; 3]);
    profile_subdomain_setup([30; 3], [5; 3]);
    profile_subdomain_setup([30; 3], [6; 3]);
}
