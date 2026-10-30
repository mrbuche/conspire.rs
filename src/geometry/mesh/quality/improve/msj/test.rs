use super::{Energy, Frame, Gradient, POINTS, SERIAL, scales, terms};
use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates,
        mesh::{Connectivity, Dualization, Fitting, Mesh, MeshError, Tessellation, Verdict},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::{Quantity, Scalar, Tensor, TensorListVec},
    units::{Area, Dimensionless},
};
use std::{collections::BTreeSet, f64::consts::TAU, sync::Mutex, time::Instant};

const SWEEPS: usize = 20;
const UNREACHABLE: Quantity = Dimensionless::of(1.0);

fn improve(mesh: &mut Mesh<3>, target: &Tessellation, sweeps: usize) -> MeshError {
    match mesh.improve_scaled_jacobian(target, UNREACHABLE, sweeps) {
        Err(error @ (MeshError::Budget { .. } | MeshError::Stall { .. })) => error,
        other => panic!("{other:?}"),
    }
}

type Raw = [[Scalar; 3]; 8];

const UNIT: Raw = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0],
    [0.0, 1.0, 1.0],
];

fn offset(index: usize, seed: Scalar) -> Scalar {
    (seed + 12.9898 * index as Scalar).sin() * 43758.5453 % 1.0
}

fn distorted(scale: Scalar, seed: Scalar) -> Raw {
    let mut nodes = UNIT;
    nodes
        .iter_mut()
        .enumerate()
        .for_each(|(n, node)| (0..3).for_each(|i| node[i] += scale * offset(3 * n + i, seed)));
    nodes
}

fn typed(nodes: &Raw) -> CoordinateList<3, 8> {
    nodes.iter().map(|&node| node.into()).collect()
}

fn energy(theta: Scalar, nodes: &Raw) -> Scalar {
    terms(
        Quantity::new(theta),
        Quantity::new(1.3),
        &typed(nodes),
        false,
    )
    .0
    .value()
}

fn check(theta: Scalar, nodes: Raw) {
    let (_, analytic) = terms(
        Quantity::new(theta),
        Quantity::new(1.3),
        &typed(&nodes),
        true,
    );
    let step = 1.0e-6;
    (0..8).for_each(|n| {
        (0..3).for_each(|i| {
            let mut above = nodes;
            let mut below = nodes;
            above[n][i] += step;
            below[n][i] -= step;
            let numerical = (energy(theta, &above) - energy(theta, &below)) / (2.0 * step);
            let exact = analytic[n][i].value();
            assert!(
                (exact - numerical).abs() < 1.0e-6,
                "node {n} axis {i}: {exact} vs {numerical}"
            );
        })
    })
}

#[test]
fn gradient_matches_finite_difference_on_the_scaled_branch() {
    check(2.0, distorted(0.25, 1.0))
}

#[test]
fn gradient_matches_finite_difference_on_the_jacobian_branch() {
    let mut nodes = UNIT;
    nodes[6] = [0.2, 0.3, 0.4];
    nodes[2] = [0.1, 0.2, -0.3];
    let coordinates = typed(&nodes);
    assert!(
        POINTS
            .iter()
            .any(|weights| Frame::new(weights, &coordinates).scaled < Quantity::default())
    );
    check(0.5, nodes)
}

#[test]
fn rectified_branch_is_flat() {
    let (value, slope) = terms(Quantity::new(0.1), Quantity::new(1.0), &typed(&UNIT), true);
    assert_eq!(value.value(), 0.0);
    assert!(
        slope
            .iter()
            .all(|node| (0..3).all(|i| node[i].value() == 0.0))
    );
}

fn cube() -> Tessellation {
    let faces: Vec<[usize; 3]> = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ];
    Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        UNIT.to_vec().into(),
    ))
    .into()
}

fn grid(n: usize, wobble: Scalar) -> Mesh<3> {
    let id = |i: usize, j: usize, k: usize| i + (n + 1) * (j + (n + 1) * k);
    let mut coordinates = Vec::new();
    for k in 0..=n {
        for j in 0..=n {
            for i in 0..=n {
                let index = id(i, j, k);
                let mut point = [i, j, k].map(|c| c as Scalar / n as Scalar);
                (0..3).for_each(|axis| {
                    let coordinate = [i, j, k][axis];
                    if coordinate != 0 && coordinate != n {
                        point[axis] += wobble * offset(3 * index + axis, 7.0) / (3 * n) as Scalar
                    }
                });
                coordinates.push(point);
            }
        }
    }
    let mut connectivity = Vec::new();
    for k in 0..n {
        for j in 0..n {
            for i in 0..n {
                connectivity.push([
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1),
                ]);
            }
        }
    }
    (
        vec![Connectivity::Hexahedral(connectivity.into())],
        coordinates.into(),
    )
        .into()
}

fn worst(mesh: &Mesh<3>) -> Scalar {
    mesh.minimum_scaled_jacobians()
        .into_iter()
        .flatten()
        .fold(Scalar::INFINITY, Scalar::min)
}

#[test]
fn improves_a_wobbly_grid_without_leaving_the_cube() {
    let n = 5;
    let mut mesh = grid(n, 0.9);
    let before = worst(&mesh);
    improve(&mut mesh, &cube(), SWEEPS);
    let after = worst(&mesh);
    assert!(after > before + 0.05, "{before} -> {after}");
    mesh.coordinates().iter().for_each(|node| {
        assert!((0..3).all(|i| node[i].value() > -1.0e-3 && node[i].value() < 1.0 + 1.0e-3))
    });
}

#[test]
fn keeps_the_cube_corners_and_edges() {
    let mut mesh = grid(4, 0.9);
    improve(&mut mesh, &cube(), SWEEPS);
    let corners = UNIT.map(|corner| {
        mesh.coordinates()
            .iter()
            .map(|node| {
                (0..3)
                    .map(|i| (node[i].value() - corner[i]).powi(2))
                    .sum::<Scalar>()
                    .sqrt()
            })
            .fold(Scalar::INFINITY, Scalar::min)
    });
    corners
        .iter()
        .for_each(|&distance| assert!(distance < 1.0e-3, "{distance}"));
    let boundary: Vec<usize> = mesh
        .exterior_faces()
        .into_iter()
        .flatten()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    boundary.iter().for_each(|&node| {
        let flush = (0..3)
            .filter(|&i| {
                let c = mesh.coordinates()[node][i].value();
                c.abs() < 1.0e-3 || (c - 1.0).abs() < 1.0e-3
            })
            .count();
        assert!(flush >= 1);
    });
}

#[test]
fn never_lowers_the_worst_scaled_jacobian() {
    let mut mesh = grid(4, 0.5);
    let before = worst(&mesh);
    improve(&mut mesh, &cube(), SWEEPS);
    assert!(worst(&mesh) >= before);
}

#[test]
fn zero_sweeps_leave_the_mesh_unchanged() {
    let mut mesh = grid(4, 0.9);
    let before = mesh.coordinates().clone();
    improve(&mut mesh, &cube(), 0);
    assert_eq!(mesh.coordinates(), &before);
}

#[test]
fn one_sweep_stops_on_the_budget() {
    let mut mesh = grid(4, 1.5);
    let error = improve(&mut mesh, &cube(), 1);
    assert!(
        matches!(error, MeshError::Budget { sweeps: 1, theta } if (theta + 0.42).abs() < 1.0e-9)
    );
}

#[test]
fn the_full_budget_stops_on_a_stall() {
    let mut mesh = grid(4, 1.5);
    let error = improve(&mut mesh, &cube(), SWEEPS);
    assert!(
        matches!(error, MeshError::Stall { sweeps: 2, theta } if (theta + 0.42).abs() < 1.0e-9)
    );
}

#[test]
fn a_reachable_threshold_is_met() {
    let mut full = grid(4, 0.9);
    improve(&mut full, &cube(), SWEEPS);
    let threshold = worst(&full) - 0.1;
    let mut mesh = grid(4, 0.9);
    mesh.improve_scaled_jacobian(&cube(), Quantity::new(threshold), SWEEPS)
        .unwrap();
    assert!(worst(&mesh) >= threshold);
}

#[test]
fn a_mesh_already_over_the_threshold_is_returned_unchanged() {
    let mut mesh = grid(4, 0.0);
    let before = mesh.coordinates().clone();
    mesh.improve_scaled_jacobian(&cube(), Quantity::new(0.9), SWEEPS)
        .unwrap();
    assert_eq!(mesh.coordinates(), &before);
}

#[test]
fn an_unreachable_threshold_is_an_error_but_keeps_the_improvement() {
    let mut mesh = grid(4, 0.9);
    let before = worst(&mesh);
    improve(&mut mesh, &cube(), SWEEPS);
    assert!(worst(&mesh) > before + 0.05);
}

fn torus(major: Scalar, minor: Scalar, around: usize, tube: usize) -> Tessellation {
    let mut coordinates = Vec::new();
    (0..around).for_each(|i| {
        let theta = TAU * i as Scalar / around as Scalar;
        (0..tube).for_each(|j| {
            let phi = TAU * j as Scalar / tube as Scalar;
            let radius = major + minor * phi.cos();
            coordinates.push([
                radius * theta.cos(),
                radius * theta.sin(),
                minor * phi.sin(),
            ])
        })
    });
    let index = |i: usize, j: usize| (i % around) * tube + (j % tube);
    let faces: Vec<[usize; 3]> = (0..around)
        .flat_map(|i| {
            (0..tube).flat_map(move |j| {
                [
                    [index(i, j), index(i + 1, j), index(i + 1, j + 1)],
                    [index(i, j), index(i + 1, j + 1), index(i, j + 1)],
                ]
            })
        })
        .collect();
    Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        coordinates.into(),
    ))
    .into()
}

fn gap(mesh: &Mesh<3>, target: &Tessellation) -> Scalar {
    let surface = target.mesh().coordinates();
    let elements: Vec<&[usize]> = target.mesh().connectivities().iter().flatten().collect();
    let boundary: BTreeSet<usize> = mesh.exterior_faces().into_iter().flatten().collect();
    boundary
        .into_iter()
        .map(|node| {
            let point = &mesh.coordinates()[node];
            let (closest, _) = target
                .bvh()
                .closest_point(
                    &Coordinate::from([point[0].value(), point[1].value(), point[2].value()]),
                    surface,
                    &elements,
                )
                .unwrap();
            (0..3)
                .map(|i| (closest[i].value() - point[i].value()).powi(2))
                .sum::<Scalar>()
                .sqrt()
        })
        .fold(0.0, Scalar::max)
}

#[test]
fn improves_a_buffered_torus() {
    let scale = 1.0;
    let tessellation = torus(scale, 0.15 * scale, 64, 24);
    let mut octree =
        Octree::<u16, usize>::from_features(&tessellation, 3.0, CurvatureSizing::default(), 0)
            .unwrap();
    octree
        .equilibrate(Balancing::Strong(1), Pairing::Regular)
        .unwrap();
    let mut mesh = octree.dualize();
    tessellation.trim(&mut mesh).unwrap();
    let mut mesh = mesh.buffer(&tessellation, Fitting::Snap).unwrap();
    let (before, before_gap) = (worst(&mesh), gap(&mesh, &tessellation));
    let start = Instant::now();
    improve(&mut mesh, &tessellation, SWEEPS);
    let (after, after_gap) = (worst(&mesh), gap(&mesh, &tessellation));
    eprintln!(
        "BENCH hexes {} sj {before:.4} -> {after:.4} gap {before_gap:.2e} -> {after_gap:.2e} in {:?}",
        mesh.number_of_elements(),
        start.elapsed()
    );
    assert!(after >= before);
}

#[test]
fn parallel_evaluation_matches_the_serial_sum_exactly() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    assert!(hexes.len() > 2 * SERIAL);
    let coordinates = mesh.coordinates().clone();
    let scales = scales(&hexes, &coordinates);
    let theta = Quantity::new(0.3);
    let mut total = Quantity::<Area>::default();
    let mut gradient = Gradient::zero(coordinates.len());
    hexes.iter().zip(&scales).for_each(|(hex, &scale)| {
        let nodes = hex.iter().map(|&node| coordinates[node].clone()).collect();
        let (value, slopes) = terms(theta, scale, &nodes, true);
        total -= value;
        hex.iter()
            .zip(slopes)
            .for_each(|(&node, slope)| gradient[node] -= slope)
    });
    let energy = Energy {
        boundary: &[],
        hexes: &hexes,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales,
        scratch: Mutex::default(),
        theta,
    };
    let mut parallel = Gradient::zero(coordinates.len());
    let value = energy.evaluate(&coordinates, Some(&mut parallel));
    assert_eq!(value.value().to_bits(), total.value().to_bits());
    gradient.iter().zip(parallel.iter()).for_each(|(a, b)| {
        (0..3).for_each(|i| assert_eq!(a[i].value().to_bits(), b[i].value().to_bits()))
    });
    assert_eq!(
        energy.evaluate(&coordinates, None).value().to_bits(),
        total.value().to_bits()
    );
}

#[test]
fn evaluation_keeps_nothing_from_the_previous_call() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    assert!(hexes.len() > 2 * SERIAL);
    let coordinates = mesh.coordinates().clone();
    let mut stretched = coordinates.clone();
    stretched.iter_mut().for_each(|node| node[0] *= 1.1);
    let scales = scales(&hexes, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales: scales.clone(),
        scratch: Mutex::default(),
        theta: Quantity::new(0.3),
    };
    let gradient = |energy: &Energy, at: &Coordinates<3>| {
        let mut slopes = Gradient::zero(at.len());
        let value = energy.evaluate(at, Some(&mut slopes));
        (value.value().to_bits(), slopes)
    };
    let value = |energy: &Energy, at: &Coordinates<3>| energy.evaluate(at, None).value().to_bits();
    let (fresh_a, fresh_b) = (energy(), energy());
    let reference_a = gradient(&fresh_a, &coordinates);
    let reference_b = gradient(&fresh_b, &stretched);
    assert_ne!(reference_a.0, reference_b.0);
    let reused = energy();
    for (at, reference) in [
        (&coordinates, &reference_a),
        (&stretched, &reference_b),
        (&coordinates, &reference_a),
    ] {
        let (bits, slopes) = gradient(&reused, at);
        assert_eq!(bits, reference.0);
        slopes.iter().zip(reference.1.iter()).for_each(|(a, b)| {
            (0..3).for_each(|i| assert_eq!(a[i].value().to_bits(), b[i].value().to_bits()))
        });
    }
    assert_eq!(value(&reused, &stretched), reference_b.0);
    assert_eq!(value(&reused, &coordinates), reference_a.0);
    assert_eq!(value(&reused, &stretched), reference_b.0);
}
