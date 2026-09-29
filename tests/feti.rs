#![cfg(feature = "fem")]

use conspire::{
    constitutive::solid::{elastic::AlmansiHamelLagrangian, hyperelastic::NeoHookean},
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, SecondOrderMinimize,
        block::{Block, element::linear::Hexahedron},
    },
    feti::{Feti, GMRES},
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh, Partition},
    },
    math::{
        Matrix, Tensor, Vector,
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
            .map(|&point| perturbed(point, nel[0] as f64))
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

fn multiplier_problem(nel: [usize; 3]) -> (Model<HexBlock, 3>, EqualityConstraint) {
    let (model, _) = problem(nel);
    let (_, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let reference: Vec<[f64; 3]> = coordinates
        .iter()
        .map(|c| [c[0].value(), c[1].value(), c[2].value()])
        .collect();
    let length = nel[0] as f64;
    let mut rows: Vec<(Vec<(usize, f64)>, f64)> = Vec::new();
    reference.iter().enumerate().for_each(|(node, point)| {
        if point[0].abs() < 1e-9 {
            (0..3).for_each(|component| {
                rows.push((vec![(3 * node + component, 1.0)], point[component]))
            })
        } else if (point[0] - length).abs() < 1e-9 {
            rows.push((vec![(3 * node, 1.0)], 1.1 * length))
        }
    });
    let far: Vec<usize> = (0..reference.len())
        .filter(|&node| (reference[node][0] - length).abs() < 1e-9)
        .collect();
    let (a, b) = (far[0], far[far.len() - 1]);
    rows.push((
        vec![(3 * a + 2, 1.0), (3 * b + 2, -1.0)],
        reference[a][2] - reference[b][2],
    ));
    let mut matrix = Matrix::zero(rows.len(), 3 * reference.len());
    let mut vector = Vector::zero(rows.len());
    rows.iter().enumerate().for_each(|(row, (entries, value))| {
        entries
            .iter()
            .for_each(|&(dof, coefficient)| matrix[row][dof] = coefficient);
        vector[row] = *value;
    });
    (model, EqualityConstraint::Linear(matrix, vector))
}

#[test]
fn newton_with_feti_enforces_linear_constraints_by_multipliers() {
    let nel = [6; 3];
    let (model, constraint) = multiplier_problem(nel);
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
    drop(model);
    let (model, constraint) = multiplier_problem(nel);
    let decomposed = model
        .minimize(
            constraint,
            NewtonRaphson {
                abs_tol: TOLERANCES,
                linear_solver: Feti {
                    partition: mesh(nel).partition_box([2; 3]),
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap_or_else(|error| panic!("FETI solve failed: {error}"));
    let difference = largest_difference(&sparse, &decomposed);
    assert!(difference < 1e-6, "coordinates differ by {difference:e}");
}

#[test]
fn prescribed_displacements_by_multipliers_are_met_and_zero_ones_match_elimination() {
    use conspire::feti::BoundaryConditions;
    let nel = [6; 3];
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let block = block(connectivities, &coordinates, 1.0);
    let current = NodalCoordinates::from(
        coordinates
            .iter()
            .map(|c| perturbed([c[0].value(), c[1].value(), c[2].value()], 6.0))
            .collect::<Vec<_>>(),
    );
    let on_plane = |plane: f64| -> Vec<usize> {
        (0..coordinates.len())
            .filter(|&node| (coordinates[node][0].value() - plane).abs() < 1e-9)
            .collect()
    };
    let feti = Feti {
        partition: mesh(nel).partition_box([2; 3]),
        ..Default::default()
    };
    let eliminated = feti
        .solve(
            &block,
            &current,
            &BoundaryConditions::new(
                on_plane(0.0)
                    .into_iter()
                    .flat_map(|node| (0..3).map(move |component| (node, component)))
                    .collect(),
            ),
        )
        .unwrap_or_else(|error| panic!("{error}"));
    let pinned = on_plane(0.0)
        .into_iter()
        .flat_map(|node| (0..3).map(move |component| (node, component)))
        .fold(
            BoundaryConditions::none(),
            |conditions, (node, component)| conditions.prescribed(node, component, 0.0),
        );
    let (multiplied, _) = feti
        .solve_constrained(&block, &current, &pinned)
        .unwrap_or_else(|error| panic!("{error}"));
    let scale = eliminated.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let difference = eliminated
        .iter()
        .zip(multiplied.iter())
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        difference < 1e-8 * scale,
        "differ by {difference:e} of {scale:e}"
    );
    let stretched = on_plane(6.0).into_iter().fold(pinned, |conditions, node| {
        conditions.prescribed(node, 0, 0.25)
    });
    let (stretch, multipliers) = feti
        .solve_constrained(&block, &current, &stretched)
        .unwrap_or_else(|error| panic!("{error}"));
    on_plane(6.0)
        .into_iter()
        .for_each(|node| assert!((stretch[3 * node] - 0.25).abs() < 1e-8));
    assert!(multipliers.iter().any(|&force| force.abs() > 1e-6));
}
