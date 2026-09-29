#![cfg(feature = "fem")]

use conspire::{
    constitutive::solid::{elastic::AlmansiHamelLagrangian, hyperelastic::NeoHookean},
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, SecondOrderMinimize,
        block::{Block, element::linear::Hexahedron},
    },
    feti::{Feti, Formulation, GMRES},
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh, Partition},
    },
    math::{
        Tensor,
        optimize::{Direct, EqualityConstraint, NewtonRaphson, Tolerances},
    },
    units::Stress,
};
use std::time::{Duration, Instant};

const TOLERANCES: Tolerances = Tolerances {
    constraint: 1e-9,
    residual: 1e-9,
};

type HexBlock = Block<NeoHookean, Hexahedron, 8, 3, 8, 8>;

fn perturbed(reference: [f64; 3], extent: f64) -> [f64; 3] {
    let [x, y, z] = reference;
    [
        x + 0.02 * y + 0.005 * y * z / extent,
        y + 0.01 * z,
        z + 0.015 * x * x / extent,
    ]
}

fn mesh(nel: [usize; 3]) -> Mesh<3> {
    Mesh::from_voxels(Voxels::new(vec![1u8; nel.iter().product()], nel), None)
}

fn block(connectivities: Connectivities, coordinates: &Coordinates<3>, scale: f64) -> HexBlock {
    let elements: Vec<[usize; 8]> = connectivities
        .into_members()
        .into_iter()
        .flat_map(|connectivity| match connectivity {
            Connectivity::Hexahedral(hexahedra) => hexahedra.into_iter().collect::<Vec<_>>(),
            _ => panic!("expected a hexahedral mesh"),
        })
        .collect();
    HexBlock::from((
        NeoHookean {
            bulk_modulus: Stress::pascals(13.0 * scale),
            shear_modulus: Stress::pascals(3.0 * scale),
        },
        elements,
        &coordinates
            .iter()
            .map(|coordinate| coordinate.clone().with_unit())
            .collect(),
    ))
}

fn problem(nel: [usize; 3]) -> (Model<HexBlock, 3>, EqualityConstraint) {
    scaled_problem(nel, 1.0, &[0.0])
}

fn scaled_problem(
    nel: [usize; 3],
    scale: f64,
    fixed_planes: &[f64],
) -> (Model<HexBlock, 3>, EqualityConstraint) {
    strained_problem(nel, scale, fixed_planes, 1.0)
}

fn strained_problem(
    nel: [usize; 3],
    scale: f64,
    fixed_planes: &[f64],
    strain: f64,
) -> (Model<HexBlock, 3>, EqualityConstraint) {
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let reference: Vec<[f64; 3]> = coordinates
        .iter()
        .map(|c| [c[0].value(), c[1].value(), c[2].value()])
        .collect();
    let fixed = reference
        .iter()
        .enumerate()
        .filter(|(_, point)| {
            fixed_planes
                .iter()
                .any(|&plane| (point[0] - plane).abs() < 1e-9)
        })
        .flat_map(|(node, _)| (0..3).map(move |component| 3 * node + component))
        .collect();
    let start = NodalReferenceCoordinates::from(
        reference
            .iter()
            .map(|&point| {
                let moved = perturbed(point, nel[0] as f64);
                if strain == 1.0 {
                    moved
                } else {
                    std::array::from_fn(|axis| point[axis] + strain * (moved[axis] - point[axis]))
                }
            })
            .collect::<Vec<_>>(),
    );
    (
        Model::from((block(connectivities, &coordinates, scale), start)),
        EqualityConstraint::Fixed(fixed),
    )
}

fn largest_difference(a: &NodalCoordinates<3>, b: &NodalCoordinates<3>) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i].value() - b[i].value()).abs()))
        .fold(0.0, f64::max)
}

fn compare(nel: [usize; 3], divisions: [usize; 3]) -> (f64, Duration, Duration) {
    let (model, constraint) = problem(nel);
    let clock = Instant::now();
    let sparse = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Direct,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("sparse solve failed: {error}"));
    let sparse_time = clock.elapsed();
    drop(model);
    let (model, constraint) = problem(nel);
    let feti = NewtonRaphson {
        linear_solver: Feti {
            partition: mesh(nel).partition_box(divisions),
            ..Default::default()
        },
        abs_tol: TOLERANCES,
        ..Default::default()
    };
    let clock = Instant::now();
    let decomposed = model
        .minimize(constraint, feti)
        .unwrap_or_else(|error| panic!("FETI solve failed: {error}"));
    let feti_time = clock.elapsed();
    (
        largest_difference(&sparse, &decomposed),
        sparse_time,
        feti_time,
    )
}

#[test]
fn newton_with_feti_matches_newton_with_the_sparse_solve() {
    let (difference, ..) = compare([6; 3], [2; 3]);
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

#[test]
fn root_finding_of_an_elastic_model_with_feti_matches_the_sparse_solve() {
    use conspire::fem::FirstOrderRoot;
    let (model, constraint) = problem([6; 3]);
    let sparse = model
        .root(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Direct,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("sparse root failed: {error}"));
    drop(model);
    let (model, constraint) = problem([6; 3]);
    let decomposed = model
        .root(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: mesh([6; 3]).partition_box([2; 3]),
                    method: GMRES,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("FETI root failed: {error}"));
    let difference = largest_difference(&sparse, &decomposed);
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

type NonsymmetricBlock = Block<AlmansiHamelLagrangian, Hexahedron, 8, 3, 8, 8>;

fn nonsymmetric_problem(nel: [usize; 3]) -> (Model<NonsymmetricBlock, 3>, EqualityConstraint) {
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let reference: Vec<[f64; 3]> = coordinates
        .iter()
        .map(|c| [c[0].value(), c[1].value(), c[2].value()])
        .collect();
    let fixed = reference
        .iter()
        .enumerate()
        .filter(|(_, point)| point[0].abs() < 1e-9)
        .flat_map(|(node, _)| (0..3).map(move |component| 3 * node + component))
        .collect();
    let elements: Vec<[usize; 8]> = connectivities
        .into_members()
        .into_iter()
        .flat_map(|connectivity| match connectivity {
            Connectivity::Hexahedral(hexahedra) => hexahedra.into_iter().collect::<Vec<_>>(),
            _ => panic!("expected a hexahedral mesh"),
        })
        .collect();
    let block = NonsymmetricBlock::from((
        AlmansiHamelLagrangian {
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        elements,
        &coordinates
            .iter()
            .map(|coordinate| coordinate.clone().with_unit())
            .collect(),
    ));
    let start = NodalReferenceCoordinates::from(
        reference
            .iter()
            .map(|&point| perturbed(point, nel[0] as f64))
            .collect::<Vec<_>>(),
    );
    (
        Model::from((block, start)),
        EqualityConstraint::Fixed(fixed),
    )
}

#[test]
fn feti_solves_a_model_with_a_nonsymmetric_tangent() {
    use conspire::fem::FirstOrderRoot;
    let nel = [6; 3];
    let (model, constraint) = nonsymmetric_problem(nel);
    let sparse = model
        .root(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Direct,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("sparse root failed: {error}"));
    drop(model);
    let (model, constraint) = nonsymmetric_problem(nel);
    let decomposed = model
        .root(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: mesh(nel).partition_box([2; 3]),
                    method: GMRES,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("FETI root failed: {error}"));
    let difference = largest_difference(&sparse, &decomposed);
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

#[test]
fn gmres_on_the_dual_problem_matches_conjugate_gradients() {
    let solve = |method| {
        let (model, constraint) = problem([6; 3]);
        model
            .minimize(
                constraint,
                NewtonRaphson {
                    abs_tol: TOLERANCES,
                    linear_solver: Feti {
                        partition: mesh([6; 3]).partition_box([2; 3]),
                        method,
                        ..Default::default()
                    },
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("FETI solve failed: {error}"))
    };
    let difference = largest_difference(&solve(GMRES), &solve(Default::default()));
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

#[test]
#[ignore]
fn benchmark_scaling_24() {
    let (difference, sparse, feti) = compare([24; 3], [4; 3]);
    println!(
        "[24, 24, 24]: Newton on the sparse solve {:.0} ms, on FETI-DP in [4, 4, 4] parts {:.0} \
         ms; coordinates differ by {difference:e}",
        sparse.as_secs_f64() * 1e3,
        feti.as_secs_f64() * 1e3,
    );
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

#[test]
fn the_linear_solver_path_reproduces_the_block_solve() {
    use conspire::{
        feti::{BoundaryConditions, DecomposableElements},
        math::{Vector, optimize::LinearSolver},
    };
    let nel = [6; 3];
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let block = block(connectivities, &coordinates, 1.0);
    let current = NodalCoordinates::from(
        coordinates
            .iter()
            .map(|c| perturbed([c[0].value(), c[1].value(), c[2].value()], 6.0))
            .collect::<Vec<_>>(),
    );
    let fixed: Vec<usize> = coordinates
        .iter()
        .enumerate()
        .filter(|(_, c)| c[0].value().abs() < 1e-9)
        .flat_map(|(node, _)| (0..3).map(move |component| 3 * node + component))
        .collect();
    let feti = Feti {
        partition: mesh(nel).partition_box([2; 3]),
        ..Default::default()
    };
    let expected = feti
        .solve(
            &block,
            &current,
            &BoundaryConditions::new(fixed.iter().map(|&dof| (dof / 3, dof % 3)).collect()),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    let retained: Vec<usize> = (0..3 * coordinates.len())
        .filter(|dof| !fixed.contains(dof))
        .collect();
    let solution = LinearSolver::solve(
        &feti,
        block.element_systems(&current).unwrap(),
        &retained,
        &Vector::zero(retained.len()),
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let scale = expected.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let difference = retained
        .iter()
        .zip(solution.iter())
        .fold(0.0_f64, |m, (&dof, &v)| m.max((expected[dof] - v).abs()));
    assert!(
        difference < 1e-8 * scale,
        "differ by {difference:e} of {scale:e}"
    );
}

#[test]
fn a_partition_that_leaves_a_subdomain_floating_is_refused() {
    let nel = [4; 3];
    let (model, constraint) = problem(nel);
    let error = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: mesh(nel).partition_box([2, 1, 1]),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .err()
        .unwrap_or_else(|| panic!("a floating subdomain must be refused"));
    let message = error.to_string();
    assert!(
        message.contains("Subdomain 1 is left floating"),
        "unexpected error: {message}"
    );
}

fn refusal(
    nel: [usize; 3],
    elements_parts: Vec<usize>,
    scale: f64,
    fixed_planes: &[f64],
) -> String {
    let (model, constraint) = scaled_problem(nel, scale, fixed_planes);
    model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: Partition::new(&mesh(nel), elements_parts),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .err()
        .unwrap_or_else(|| panic!("a subdomain with a free part must be refused"))
        .to_string()
}

#[test]
fn a_subdomain_in_two_pieces_with_one_free_is_refused() {
    [1e-3, 1.0, 1e9, 1e12].into_iter().for_each(|scale| {
        let message = refusal([4, 1, 1], vec![0, 1, 1, 0], scale, &[0.0, 2.0]);
        assert!(
            message.contains("The stiffness of subdomain 0 is singular"),
            "scale {scale:e}: {message}"
        );
    })
}

#[test]
fn a_subdomain_hinged_about_an_edge_is_refused() {
    [1e-3, 1.0, 1e9, 1e12].into_iter().for_each(|scale| {
        let message = refusal([2, 2, 1], vec![0, 1, 1, 0], scale, &[0.0]);
        assert!(
            message.contains("is singular") || message.contains("not positive definite"),
            "scale {scale:e}: {message}"
        );
    })
}

#[test]
#[ignore]
fn benchmark_cg_vs_gmres() {
    use conspire::math::optimize::KrylovMethod;
    for (nel, divisions) in [([24; 3], [4; 3]), ([24; 3], [6; 3])] {
        for method in [
            KrylovMethod::ConjugateGradients,
            GMRES,
            KrylovMethod::ConjugateGradients,
            GMRES,
        ] {
            let (model, constraint) = problem(nel);
            let clock = Instant::now();
            model
                .minimize(
                    constraint,
                    NewtonRaphson {
                        abs_tol: TOLERANCES,
                        linear_solver: Feti {
                            partition: mesh(nel).partition_box(divisions),
                            method,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap_or_else(|error| panic!("FETI solve failed: {error}"));
            eprintln!(
                "BENCH {nel:?} in {divisions:?} parts {method:?}: Newton total {:.0} ms",
                clock.elapsed().as_secs_f64() * 1e3
            );
        }
    }
}

#[test]
fn newton_with_classical_feti_matches_newton_with_the_sparse_solve() {
    let nel = [6; 3];
    let (model, constraint) = problem(nel);
    let sparse = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Direct,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("sparse solve failed: {error}"));
    let (model, constraint) = problem(nel);
    let classical = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: mesh(nel).partition_box([2; 3]),
                    formulation: Formulation::Classical,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("classical FETI solve failed: {error}"));
    let difference = largest_difference(&sparse, &classical);
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

#[test]
fn classical_differs_from_dual_primal_by_the_prestress_and_takes_more_iterations() {
    use conspire::feti::{BoundaryConditions, Preconditioner};
    let nel = [8; 3];
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let block = block(connectivities, &coordinates, 1.0);
    let boundary_conditions = BoundaryConditions::new(
        coordinates
            .iter()
            .enumerate()
            .filter(|(_, c)| c[0].value().abs() < 1e-9)
            .flat_map(|(node, _)| (0..3).map(move |component| (node, component)))
            .collect(),
    );
    [1.0, 0.1, 0.01].into_iter().for_each(|strain| {
        let (model, constraint) = strained_problem(nel, 1.0, &[0.0], strain);
        let converged = model
            .minimize(
                constraint,
                NewtonRaphson {
                    abs_tol: TOLERANCES,
                    linear_solver: Direct,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|error| panic!("sparse solve failed: {error}"));
        let current = NodalCoordinates::from(
            converged
                .iter()
                .enumerate()
                .map(|(node, c)| {
                    std::array::from_fn(|axis| {
                        c[axis].value() + 1e-7 * (((7919 * node + 104729 * axis) % 13) as f64 - 6.0)
                    })
                })
                .collect::<Vec<[f64; 3]>>(),
        );
        [[2; 3], [4; 3]].into_iter().for_each(|divisions| {
            let solve_with = |formulation, preconditioner| {
                Feti {
                    partition: mesh(nel).partition_box(divisions),
                    formulation,
                    preconditioner,
                    rel_tol: 1e-12,
                    ..Default::default()
                }
                .solve_with_statistics(&block, &current, &boundary_conditions)
                .unwrap_or_else(|error| panic!("{error}"))
            };
            let solve = |formulation| solve_with(formulation, Preconditioner::Dirichlet);
            let (dual_primal, dual_primal_statistics) = solve(Formulation::DualPrimal);
            let (classical, classical_statistics) = solve(Formulation::Classical);
            [Formulation::DualPrimal, Formulation::Classical]
                .into_iter()
                .for_each(|formulation| {
                    let (unscaled, _) = solve(formulation);
                    let (scaled, _) = solve_with(formulation, Preconditioner::ScaledDirichlet);
                    let size = unscaled.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
                    let gap = unscaled
                        .iter()
                        .zip(scaled.iter())
                        .fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()));
                    assert!(
                        gap < 1e-6 * size,
                        "{formulation:?}: scaled differs by {gap:e} of {size:e}"
                    );
                });
            let dual_primal_applications = dual_primal_statistics.dual_applications;
            let classical_applications = classical_statistics.dual_applications;
            let scale = dual_primal.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
            let difference = dual_primal
                .iter()
                .zip(classical.iter())
                .fold(0.0_f64, |m, (&a, &b)| m.max((a - b).abs()));
            println!(
                "strain {strain}, divisions {divisions:?}: dual-primal \
                {dual_primal_applications} applications, classical {classical_applications}, \
                relative difference {:e}",
                difference / scale
            );
            assert!(
                difference < 1e-2 * strain * scale,
                "differ by {difference:e} of {scale:e} at strain {strain}"
            );
            assert!(dual_primal_applications < classical_applications);
        });
    });
}

/// Run with
/// `cargo test -F fem --profile release-dev --test feti -- --ignored --nocapture benchmark`.
#[test]
#[ignore = "a benchmark, not a check"]
fn benchmark_classical_against_dual_primal() {
    use conspire::feti::{BoundaryConditions, Preconditioner, SolveStatistics};
    const REPEATS: usize = 3;
    let nel = [12; 3];
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let block = block(connectivities, &coordinates, 1.0);
    let boundary_conditions = BoundaryConditions::new(
        coordinates
            .iter()
            .enumerate()
            .filter(|(_, c)| c[0].value().abs() < 1e-9)
            .flat_map(|(node, _)| (0..3).map(move |component| (node, component)))
            .collect(),
    );
    let (model, constraint) = strained_problem(nel, 1.0, &[0.0], 0.01);
    let converged = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Direct,
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("sparse solve failed: {error}"));
    let current = NodalCoordinates::from(
        converged
            .iter()
            .enumerate()
            .map(|(node, c)| {
                std::array::from_fn(|axis| {
                    c[axis].value() + 1e-7 * (((7919 * node + 104729 * axis) % 13) as f64 - 6.0)
                })
            })
            .collect::<Vec<[f64; 3]>>(),
    );
    let milliseconds = |duration: Duration| duration.as_secs_f64() * 1e3;
    println!(
        "{} elements, best of {REPEATS}, one thread; times in ms",
        nel.iter().product::<usize>()
    );
    println!(
        "{:>8} {:>11} {:>16} {:>8} {:>9} {:>9} {:>9} {:>9}",
        "per side", "method", "preconditioner", "applied", "assembly", "setup", "solve", "total"
    );
    [2, 3, 4, 6].into_iter().for_each(|side| {
        [Formulation::DualPrimal, Formulation::Classical]
            .into_iter()
            .flat_map(|formulation| {
                [Preconditioner::Dirichlet, Preconditioner::ScaledDirichlet]
                    .map(|preconditioner| (formulation, preconditioner))
            })
            .for_each(|(formulation, preconditioner)| {
                let feti = Feti {
                    partition: mesh(nel).partition_box([side; 3]),
                    formulation,
                    preconditioner,
                    ..Default::default()
                };
                let runs: Vec<SolveStatistics> = (0..REPEATS)
                    .map(|_| {
                        feti.solve_with_statistics(&block, &current, &boundary_conditions)
                            .unwrap_or_else(|error| panic!("{error}"))
                            .1
                    })
                    .collect();
                let best = |phase: fn(&SolveStatistics) -> Duration| {
                    milliseconds(runs.iter().map(phase).min().unwrap())
                };
                let (assembly, setup, solve) = (
                    best(|s| s.assembly),
                    best(|s| s.setup),
                    best(|s| s.dual_solve),
                );
                println!(
                    "{:>8} {:>11} {:>16} {:>8} {:>9.1} {:>9.1} {:>9.1} {:>9.1}",
                    format!("{side}^3"),
                    format!("{formulation:?}"),
                    format!("{preconditioner:?}"),
                    runs[0].dual_applications,
                    assembly,
                    setup,
                    solve,
                    assembly + setup + solve,
                );
            });
    });
}
