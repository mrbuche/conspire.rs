use super::{
    Energy, Frame, Gradient, KINDS, POINTS, Reach, SERIAL, Surface, bounded, displacements,
    node_scales, scales, terms,
};
use crate::{
    geometry::{
        Coordinate, CoordinateList, Coordinates,
        mesh::{Connectivity, Dualization, Fitting, Mesh, MeshError, Tessellation, Verdict},
        ntree::{Balance, Balancing, CurvatureSizing, Octree, Pairing},
    },
    math::{
        Quantity, QuantityVector, Scalar, Tensor, TensorListVec,
        assert::{Assert, AssertionError},
    },
    units::{Area, Dimensionless, Length},
};
use std::{
    array::from_fn,
    collections::BTreeSet,
    f64::consts::{SQRT_2, TAU},
    sync::Mutex,
    time::Instant,
};

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
            .any(|weights| Frame::new(weights, &coordinates, 1.0).scaled < Quantity::default())
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
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
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
        kinds: &kinds,
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
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
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

#[test]
fn skipping_hexes_that_cannot_reach_the_threshold_changes_nothing() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let coordinates = mesh.coordinates().clone();
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales: scales.clone(),
        scratch: Mutex::default(),
        theta: Quantity::new(0.3),
    };
    let reused = energy();
    reused.evaluate(&coordinates, Some(&mut Gradient::zero(coordinates.len())));
    for (round, amplitude) in [1.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2]
        .into_iter()
        .enumerate()
    {
        let mut moved = coordinates.clone();
        moved.iter_mut().enumerate().for_each(|(n, node)| {
            (0..3).for_each(|i| {
                node[i] += Quantity::new(amplitude * offset(3 * n + i, round as Scalar))
            })
        });
        assert_eq!(
            reused.evaluate(&moved, None).value().to_bits(),
            energy().evaluate(&moved, None).value().to_bits(),
            "amplitude {amplitude}"
        );
    }
}

#[test]
fn skipping_hexes_in_the_gradient_changes_nothing_over_many_moves() {
    let mesh = grid(11, 0.9);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let mut coordinates = mesh.coordinates().clone();
    let kinds = vec![0u8; hexes.len()];
    let scales = scales(&hexes, &kinds, &coordinates);
    let energy = || Energy {
        boundary: &[],
        hexes: &hexes,
        kinds: &kinds,
        lambda: TensorListVec::default(),
        planes: Vec::new(),
        rho: Quantity::new(1.0),
        scales: scales.clone(),
        scratch: Mutex::default(),
        theta: Quantity::new(0.6),
    };
    let gradient = |energy: &Energy, at: &Coordinates<3>| {
        let mut slopes = Gradient::zero(at.len());
        let value = energy.evaluate(at, Some(&mut slopes));
        (value.value().to_bits(), slopes)
    };
    let reused = energy();
    gradient(&reused, &coordinates);
    for round in 0..40 {
        coordinates.iter_mut().enumerate().for_each(|(n, node)| {
            let height = node[1];
            node[0] += height * 0.008;
            (0..3)
                .for_each(|i| node[i] += Quantity::new(1.0e-5 * offset(3 * n + i, round as Scalar)))
        });
        let (value, slopes) = gradient(&reused, &coordinates);
        let (fresh_value, fresh_slopes) = gradient(&energy(), &coordinates);
        assert_eq!(value, fresh_value, "value in round {round}");
        slopes.iter().zip(fresh_slopes.iter()).for_each(|(a, b)| {
            (0..3).for_each(|i| {
                assert_eq!(
                    a[i].value().to_bits(),
                    b[i].value().to_bits(),
                    "round {round}"
                )
            })
        });
        let mut trial = coordinates.clone();
        trial
            .iter_mut()
            .for_each(|node| node[0] += Quantity::new(1.0e-4));
        assert_eq!(
            reused.evaluate(&trial, None).value().to_bits(),
            energy().evaluate(&trial, None).value().to_bits(),
            "trial in round {round}"
        );
    }
}

#[test]
fn the_decayed_bound_never_exceeds_what_the_moved_hexahedron_still_has() {
    let nodes = distorted(0.15, 2.0);
    let (sj, _, _) = {
        let (_, _, (margin, shortest)) = bounded(
            &KINDS[0],
            Quantity::new(0.0),
            Quantity::new(1.0),
            &typed(&nodes),
            true,
        );
        (margin.value(), shortest.value(), ())
    };
    assert!(sj > 0.3);
    let theta = Quantity::new(sj - 0.2);
    let (_, _, bound) = bounded(&KINDS[0], theta, Quantity::new(1.0), &typed(&nodes), true);
    let base: Coordinates<3> = nodes.to_vec().into();
    let bounds = [bound];
    let hex = [0, 1, 2, 3, 4, 5, 6, 7];
    let (margin, shortest) = bound;
    let step = (margin - Quantity::new(1.0e-9)) * shortest / 12.0 * 0.9;
    let mut skipped = 0;
    for seed in 0..200 {
        let mut moved = nodes;
        moved.iter_mut().enumerate().for_each(|(n, node)| {
            (0..3).for_each(|i| {
                let sign = if offset(3 * n + i, seed as Scalar) < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                node[i] += sign * step.value() / 3.0_f64.sqrt()
            })
        });
        let current: Coordinates<3> = moved.to_vec().into();
        let mut displaced = Vec::new();
        displacements(&current, &base, &mut displaced);
        let reach = Reach {
            bounds: &bounds,
            moved: &displaced,
        };
        let Some(distance) = reach.within(0, &hex) else {
            continue;
        };
        skipped += 1;
        let (lower, shorter) = reach.decayed(0, distance);
        let (_, _, (margin, shortest)) =
            bounded(&KINDS[0], theta, Quantity::new(1.0), &typed(&moved), true);
        assert!(margin >= lower && shortest >= shorter, "seed {seed}");
    }
    assert_eq!(skipped, 200);
}

#[test]
fn planes_do_not_depend_on_how_the_nodes_are_shared_among_threads() -> Result<(), AssertionError> {
    let mesh = grid(15, 0.5);
    let hexes: Vec<[usize; 8]> = mesh
        .iter()
        .flat_map(|block| match block {
            Connectivity::Hexahedral(elements) => elements.iter().copied().collect::<Vec<_>>(),
            _ => unreachable!(),
        })
        .collect();
    let mut boundary: Vec<usize> = mesh.exterior_faces().into_iter().flatten().collect();
    boundary.sort_unstable();
    boundary.dedup();
    assert!(boundary.len() > 2 * SERIAL);
    let coordinates = mesh.coordinates().clone();
    let sizes = node_scales(
        &boundary,
        &hexes,
        &scales(&hexes, &vec![0u8; hexes.len()], &coordinates),
        coordinates.len(),
    );
    let target = cube();
    let surface = Surface::new(&target);
    let planes = surface.planes(&boundary, &sizes, &coordinates);
    let mut expected = Vec::new();
    for (index, &node) in boundary.iter().enumerate() {
        for mut plane in surface.planes(&[node], &sizes, &coordinates) {
            plane.boundary = index;
            expected.push(plane)
        }
    }
    assert!(!planes.is_empty());
    assert_eq!(planes.len(), expected.len());
    for (plane, expected) in planes.iter().zip(&expected) {
        assert_eq!(
            (plane.boundary, plane.slot),
            (expected.boundary, expected.slot)
        );
        Assert::eq(plane.offset.value(), expected.offset.value())?;
        Assert::eq(&plane.normal, &expected.normal)?
    }
    Ok(())
}

#[test]
fn a_node_beside_an_edge_keeps_the_planes_of_both_faces() -> Result<(), AssertionError> {
    let target = cube();
    let surface = Surface::new(&target);
    let coordinates = Coordinates::from(vec![[1.005, -2.0e-5, 0.5]]);
    let sizes: QuantityVector<Length> = [Quantity::new(0.33)].into_iter().collect();
    let planes = surface.planes(&[0], &sizes, &coordinates);
    assert_eq!(planes.len(), 2);
    let has = |normal: [Scalar; 3], offset: Scalar| {
        planes.iter().any(|plane| {
            (0..3).all(|i| (plane.normal[i].value() - normal[i]).abs() < 1.0e-12)
                && (plane.offset.value() - offset).abs() < 1.0e-12
        })
    };
    assert!(has([1.0, 0.0, 0.0], 1.0));
    assert!(has([0.0, -1.0, 0.0], 0.0));
    Ok(())
}

type Shape = fn(Scalar) -> [[Scalar; 3]; 4];

fn tetrahedron(nodes: [[Scalar; 3]; 4]) -> CoordinateList<3, 8> {
    typed(&from_fn(|i| nodes[if i < 4 { i } else { 0 }]))
}

fn tetrahedron_value(theta: Scalar, scale: Scalar, nodes: [[Scalar; 3]; 4]) -> Scalar {
    bounded(
        &KINDS[1],
        Quantity::new(theta),
        Quantity::new(scale),
        &tetrahedron(nodes),
        false,
    )
    .0
    .value()
}

fn distorted_tetrahedron(scale: Scalar, seed: Scalar) -> [[Scalar; 3]; 4] {
    let mut nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    nodes
        .iter_mut()
        .enumerate()
        .for_each(|(n, node)| (0..3).for_each(|i| node[i] += scale * offset(3 * n + i, seed)));
    nodes
}

fn mean_edge(nodes: &[[Scalar; 3]; 4]) -> Scalar {
    let distance = |a: usize, b: usize| {
        (0..3)
            .map(|i| (nodes[a][i] - nodes[b][i]).powi(2))
            .sum::<Scalar>()
            .sqrt()
    };
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        .into_iter()
        .map(|(a, b)| distance(a, b))
        .sum::<Scalar>()
        / 6.0
}

fn tetrahedron_check(theta: Scalar, nodes: [[Scalar; 3]; 4]) {
    let (_, analytic, _) = bounded(
        &KINDS[1],
        Quantity::new(theta),
        Quantity::new(1.3),
        &tetrahedron(nodes),
        true,
    );
    let step = 1.0e-6;
    (0..8).for_each(|n| {
        (0..3).for_each(|i| {
            let numerical = if n < 4 {
                let (mut above, mut below) = (nodes, nodes);
                above[n][i] += step;
                below[n][i] -= step;
                (tetrahedron_value(theta, 1.3, above) - tetrahedron_value(theta, 1.3, below))
                    / (2.0 * step)
            } else {
                0.0
            };
            let exact = analytic[n][i].value();
            assert!(
                (exact - numerical).abs() < 1.0e-6,
                "node {n} axis {i}: {exact} vs {numerical}"
            );
        })
    })
}

#[test]
fn tetrahedron_gradient_matches_finite_difference_on_the_scaled_branch() {
    tetrahedron_check(2.0, distorted_tetrahedron(0.2, 1.0))
}

#[test]
fn tetrahedron_gradient_matches_finite_difference_on_the_jacobian_branch() {
    let mut nodes = distorted_tetrahedron(0.1, 3.0);
    nodes[3][2] = -0.4;
    let frames = KINDS[1]
        .points
        .iter()
        .map(|point| Frame::new(point, &tetrahedron(nodes), KINDS[1].normalization).scaled)
        .fold(Quantity::new(Scalar::INFINITY), Quantity::min);
    assert!(frames < Quantity::default());
    tetrahedron_check(0.5, nodes)
}

#[test]
fn a_tetrahedron_scores_the_scaled_jacobian_of_the_metric() -> Result<(), AssertionError> {
    use crate::geometry::mesh::quality::metrics::tetrahedron::CORNERS;
    let determinant = |e: &[[Scalar; 3]; 3]| {
        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
            - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
            + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
    };
    let norm = |v: &[Scalar; 3]| v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    for seed in 0..12 {
        let nodes = distorted_tetrahedron(0.2, seed as Scalar);
        let reference = CORNERS
            .iter()
            .map(|(corner, adjacent)| {
                let e: [[Scalar; 3]; 3] =
                    from_fn(|k| from_fn(|i| nodes[adjacent[k]][i] - nodes[*corner][i]));
                SQRT_2 * determinant(&e) / (norm(&e[0]) * norm(&e[1]) * norm(&e[2]))
            })
            .fold(Scalar::INFINITY, Scalar::min);
        assert!(reference > 0.0);
        let scale = mean_edge(&nodes);
        let theta = 2.0;
        let value = tetrahedron_value(theta, scale, nodes);
        Assert::default().eq_within_tols(value / (scale * scale) + theta, reference)?
    }
    Ok(())
}

#[test]
fn a_regular_tetrahedron_scores_one() -> Result<(), AssertionError> {
    let nodes = [
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
    ];
    let scale = mean_edge(&nodes);
    Assert::default().eq_within_tols(
        tetrahedron_value(2.0, scale, nodes) / (scale * scale) + 2.0,
        1.0,
    )
}

#[test]
fn tetrahedron_energy_is_continuous_through_a_flat_tetrahedron() -> Result<(), AssertionError> {
    let theta = 0.3;
    let shapes: [Shape; 2] = [
        |h| {
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.75_f64.sqrt(), 0.0],
                [0.5, 0.75_f64.sqrt() / 3.0, h],
            ]
        },
        |h| {
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, h],
            ]
        },
    ];
    let delta = 1.0e-7;
    for shape in &shapes {
        let energy = |h: Scalar| {
            let nodes = shape(h);
            tetrahedron_value(theta, mean_edge(&nodes), nodes)
        };
        let right = (energy(2.0 * delta) - energy(delta)) / delta;
        let left = (energy(-delta) - energy(-2.0 * delta)) / delta;
        let flat = shape(0.0);
        let distance = |a: usize, b: usize| {
            (0..3)
                .map(|i| (flat[a][i] - flat[b][i]).powi(2))
                .sum::<Scalar>()
                .sqrt()
        };
        let product = crate::geometry::mesh::quality::metrics::tetrahedron::CORNERS
            .iter()
            .map(|(corner, adjacent)| adjacent.iter().map(|&n| distance(*corner, n)).product())
            .fold(0.0, Scalar::max);
        let tols = Assert {
            abs_tol: 1.0e-3,
            rel_tol: 1.0e-3,
            ..Default::default()
        };
        Assert {
            abs_tol: 1.0e-5,
            rel_tol: 1.0e-5,
            ..Default::default()
        }
        .eq_within_tols(energy(delta), energy(-delta))?;
        tols.eq_within_tols(right / left, mean_edge(&flat).powi(3) / product)?
    }
    Ok(())
}
