use super::{Frame, POINTS, terms};
use crate::{
    geometry::{
        Coordinate, CoordinateList,
        mesh::{Connectivity, Dualization, Fitting, Mesh, MeshError, Tessellation, Verdict},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::{Quantity, Scalar, Tensor},
    units::Dimensionless,
};
use std::{collections::BTreeSet, f64::consts::TAU, time::Instant};

const SWEEPS: usize = 20;
const UNREACHABLE: Quantity = Dimensionless::of(1.0);

fn improve(mesh: &mut Mesh<3>, target: &Tessellation, sweeps: usize) -> MeshError {
    match mesh.improve_scaled_jacobian(target, UNREACHABLE, sweeps) {
        Err(error @ (MeshError::Budget { .. } | MeshError::Stall { .. })) => error,
        other => panic!("{other:?}"),
    }
}

type Raw = [[Scalar; 3]; 8];
type Raw3 = [Scalar; 3];

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
    cube_through(|point| point)
}

fn cube_through(map: fn(Raw3) -> Raw3) -> Tessellation {
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
        UNIT.map(map).to_vec().into(),
    ))
    .into()
}

fn grid(n: usize, wobble: Scalar) -> Mesh<3> {
    grid_through(n, wobble, |point| point)
}

fn grid_through(n: usize, wobble: Scalar, map: fn(Raw3) -> Raw3) -> Mesh<3> {
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
                coordinates.push(map(point));
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
fn one_sweep_stops_the_ratchet_below_the_full_budget() {
    let mut one = grid(4, 0.9);
    improve(&mut one, &cube(), 1);
    let mut full = grid(4, 0.9);
    improve(&mut full, &cube(), SWEEPS);
    assert!(worst(&one) < worst(&full) - 0.1);
}

#[test]
fn one_sweep_stops_on_the_budget() {
    let mut mesh = grid(4, 0.9);
    let error = improve(&mut mesh, &cube(), 1);
    assert!(
        matches!(error, MeshError::Budget { sweeps: 1, theta } if (theta - 0.71).abs() < 1.0e-9)
    );
}

#[test]
fn the_full_budget_stops_on_a_stall() {
    let mut mesh = grid(4, 0.9);
    let error = improve(&mut mesh, &cube(), SWEEPS);
    assert!(
        matches!(error, MeshError::Stall { sweeps: 1, theta } if (theta - 0.85).abs() < 1.0e-9)
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

const TILT: Scalar = 0.5;

fn skew(point: Raw3) -> Raw3 {
    let sheared = [
        point[0] + 0.5 * point[1],
        point[1] + 0.3 * point[2],
        point[2],
    ];
    [
        TILT.cos() * sheared[0] - TILT.sin() * sheared[1],
        TILT.sin() * sheared[0] + TILT.cos() * sheared[1],
        sheared[2],
    ]
}

fn unskew(point: Raw3) -> Raw3 {
    let turned = [
        TILT.cos() * point[0] + TILT.sin() * point[1],
        -TILT.sin() * point[0] + TILT.cos() * point[1],
        point[2],
    ];
    let y = turned[1] - 0.3 * turned[2];
    [turned[0] - 0.5 * y, y, turned[2]]
}

#[test]
fn keeps_the_creases_of_a_skewed_box() {
    let mut mesh = grid_through(4, 0.9, skew);
    let flush = |mesh: &Mesh<3>| -> Vec<usize> {
        mesh.coordinates()
            .iter()
            .map(|node| {
                let point = unskew([node[0].value(), node[1].value(), node[2].value()]);
                point
                    .iter()
                    .filter(|c| c.abs() < 1.0e-3 || (**c - 1.0).abs() < 1.0e-3)
                    .count()
            })
            .collect()
    };
    let before = flush(&mesh);
    assert!(before.contains(&3));
    assert!(before.contains(&2));
    let worst_before = worst(&mesh);
    improve(&mut mesh, &cube_through(skew), SWEEPS);
    assert!(worst(&mesh) > worst_before + 0.02);
    let after = flush(&mesh);
    before
        .iter()
        .zip(&after)
        .enumerate()
        .for_each(|(node, (before, after))| {
            assert!(after >= before, "node {node}: {before} -> {after}")
        });
}
