use super::{Point, minimum_scaled_jacobian, terms};
use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Dualization, Fitting, Mesh, Tessellation, Verdict},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::{Scalar, Tensor},
};
use std::f64::consts::TAU;

const UNIT: [Point; 8] = [
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

fn distorted(scale: Scalar, seed: Scalar) -> [Point; 8] {
    let mut nodes = UNIT;
    nodes
        .iter_mut()
        .enumerate()
        .for_each(|(n, node)| (0..3).for_each(|i| node[i] += scale * offset(3 * n + i, seed)));
    nodes
}

fn check(theta: Scalar, nodes: [Point; 8]) {
    let (_, analytic) = terms(theta, 1.3, &nodes, true);
    let step = 1.0e-6;
    (0..8).for_each(|n| {
        (0..3).for_each(|i| {
            let mut above = nodes;
            let mut below = nodes;
            above[n][i] += step;
            below[n][i] -= step;
            let numerical = (terms(theta, 1.3, &above, false).0
                - terms(theta, 1.3, &below, false).0)
                / (2.0 * step);
            assert!(
                (analytic[n][i] - numerical).abs() < 1.0e-6,
                "node {n} axis {i}: {} vs {numerical}",
                analytic[n][i]
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
    assert!(minimum_scaled_jacobian(&[[0, 1, 2, 3, 4, 5, 6, 7]], &nodes.concat()) < 0.0);
    check(0.5, nodes)
}

#[test]
fn rectified_branch_is_flat() {
    let (value, slope) = terms(0.1, 1.0, &UNIT, true);
    assert_eq!(value, 0.1);
    assert!(slope.iter().flatten().all(|&entry| entry == 0.0));
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
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(UNIT.to_vec()),
    )))
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
    Mesh::from((
        vec![Connectivity::Hexahedral(connectivity.into())],
        coordinates.into(),
    ))
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
    mesh.improve_scaled_jacobian(&cube()).unwrap();
    let after = worst(&mesh);
    assert!(after > before + 0.05, "{before} -> {after}");
    mesh.coordinates().iter().for_each(|node| {
        assert!((0..3).all(|i| node[i].value() > -1.0e-3 && node[i].value() < 1.0 + 1.0e-3))
    });
}

#[test]
fn keeps_the_cube_corners_and_edges() {
    let mut mesh = grid(4, 0.9);
    mesh.improve_scaled_jacobian(&cube()).unwrap();
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
        .collect::<std::collections::BTreeSet<_>>()
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
    mesh.improve_scaled_jacobian(&cube()).unwrap();
    assert!(worst(&mesh) >= before);
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
    Tessellation::from(Mesh::from((
        vec![Connectivity::Triangular(faces.into())],
        Coordinates::from(coordinates),
    )))
}

fn gap(mesh: &Mesh<3>, target: &Tessellation) -> Scalar {
    let surface = target.mesh().coordinates();
    let elements: Vec<&[usize]> = target.mesh().connectivities().iter().flatten().collect();
    let boundary: std::collections::BTreeSet<usize> =
        mesh.exterior_faces().into_iter().flatten().collect();
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
    let start = std::time::Instant::now();
    mesh.improve_scaled_jacobian(&tessellation).unwrap();
    let (after, after_gap) = (worst(&mesh), gap(&mesh, &tessellation));
    eprintln!(
        "BENCH hexes {} sj {before:.4} -> {after:.4} gap {before_gap:.2e} -> {after_gap:.2e} in {:?}",
        mesh.number_of_elements(),
        start.elapsed()
    );
    assert!(after >= before);
}
