#![cfg(feature = "fem")]

use conspire::{
    constitutive::solid::hyperelastic::NeoHookean,
    fem::{
        Model, NodalCoordinates, NodalReferenceCoordinates, SecondOrderMinimizeSingle,
        block::{Block, element::linear::Hexahedron},
        feti::Feti,
    },
    geometry::{
        Coordinates,
        grid::Voxels,
        mesh::{Connectivities, Connectivity, Mesh},
    },
    math::{
        Tensor,
        optimize::{Direct, EqualityConstraint, NewtonRaphson, Tolerances},
    },
    units::Stress,
};
use std::time::{Duration, Instant};

/// FETI-DP solves iteratively, so the residual cannot be driven as far as a
/// factorization drives it.
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

fn block(connectivities: Connectivities, coordinates: &Coordinates<3>) -> HexBlock {
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
            bulk_modulus: Stress::pascals(13.0),
            shear_modulus: Stress::pascals(3.0),
        },
        elements,
        &coordinates
            .iter()
            .map(|coordinate| coordinate.clone().with_unit())
            .collect(),
    ))
}

/// The model of a hexahedral block whose x = 0 face is fixed, started from a
/// smooth non-affine configuration so that the right-hand side is nonzero
/// everywhere, and the constraint that fixes that face.
fn problem(nel: [usize; 3]) -> (Model<HexBlock, 3>, EqualityConstraint) {
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
    let start = NodalReferenceCoordinates::from(
        reference
            .iter()
            .map(|&point| perturbed(point, nel[0] as f64))
            .collect::<Vec<_>>(),
    );
    (
        Model::from((block(connectivities, &coordinates), start)),
        EqualityConstraint::Fixed(fixed),
    )
}

fn largest_difference(a: &NodalCoordinates<3>, b: &NodalCoordinates<3>) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(a, b)| (0..3).map(move |i| (a[i].value() - b[i].value()).abs()))
        .fold(0.0, f64::max)
}

/// Newton with FETI-DP as its linear solver against Newton on the sparse
/// factorization, from the same start: how far apart they end, and how long
/// each took.
fn compare(nel: [usize; 3], divisions: [usize; 3]) -> (f64, Duration, Duration) {
    let (model, constraint) = problem(nel);
    let clock = Instant::now();
    let sparse = model
        .minimize_single(
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
        .minimize_single(constraint, feti)
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

/// 24^3 hexahedra (46.9k dofs) in 4^3 = 64 subdomains of 1029 dofs. The local
/// matrices are dense.
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
        fem::feti::{BoundaryConditions, DecomposableElements},
        math::{Vector, optimize::LinearSolver},
    };
    let nel = [6; 3];
    let (connectivities, coordinates): (Connectivities, Coordinates<3>) = mesh(nel).into();
    let block = block(connectivities, &coordinates);
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
