use crate::{
    geometry::{
        Coordinate, Coordinates,
        grid::Voxels,
        mesh::{Connectivity, Mesh},
    },
    math::Quantity,
};
use std::collections::HashSet;

fn square(n: usize) -> Mesh<2> {
    let node = |i: usize, j: usize| j * (n + 1) + i;
    let coordinates: Coordinates<2> = (0..=n)
        .flat_map(|j| {
            (0..=n).map(move |i| Coordinate::from([i as f64 / n as f64, j as f64 / n as f64]))
        })
        .collect();
    let triangles: Vec<[usize; 3]> = (0..n)
        .flat_map(|j| (0..n).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            [
                [node(i, j), node(i + 1, j), node(i + 1, j + 1)],
                [node(i, j), node(i + 1, j + 1), node(i, j + 1)],
            ]
        })
        .collect();
    Mesh::from((
        vec![Connectivity::Triangular(triangles.into())],
        coordinates,
    ))
}

fn cube(n: usize) -> Mesh<3> {
    Mesh::from_voxels(Voxels::new(vec![1u8; n * n * n], [n, n, n]), None)
}

fn distance<const D: usize>(mesh: &Mesh<D>, a: usize, b: usize) -> f64 {
    let (x, y) = (&mesh.coordinates()[a], &mesh.coordinates()[b]);
    (0..D)
        .map(|k| (x[k].value() - y[k].value()).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn check<const D: usize>(mesh: &Mesh<D>, spacing: f64, seed: u64) {
    let samples = mesh.sample(Quantity::new(spacing), seed);
    samples.iter().enumerate().for_each(|(a, &i)| {
        samples[a + 1..]
            .iter()
            .for_each(|&j| assert!(distance(mesh, i, j) >= spacing, "samples too close"))
    });
    let boundary: HashSet<usize> = mesh.exterior_faces().into_iter().flatten().collect();
    let boundary_samples: Vec<usize> = samples
        .iter()
        .copied()
        .filter(|node| boundary.contains(node))
        .collect();
    assert!(
        samples[..boundary_samples.len()]
            .iter()
            .all(|node| boundary.contains(node)),
        "boundary nodes must be sampled first"
    );
    boundary.iter().for_each(|&node| {
        assert!(
            boundary_samples
                .iter()
                .any(|&sample| distance(mesh, node, sample) < spacing),
            "boundary node left uncovered by the boundary samples"
        )
    });
    (0..mesh.number_of_nodes()).for_each(|node| {
        assert!(
            samples
                .iter()
                .any(|&sample| distance(mesh, node, sample) < spacing),
            "node left uncovered"
        )
    });
}

#[test]
fn packing_2d() {
    check(&square(30), 0.1, 1);
    check(&square(30), 0.25, 2);
}

#[test]
fn packing_3d() {
    check(&cube(6), 1.5, 1);
    check(&cube(6), 2.5, 2);
}

#[test]
fn deterministic_in_seed() {
    let mesh = square(30);
    let spacing = Quantity::new(0.1);
    assert_eq!(mesh.sample(spacing, 7), mesh.sample(spacing, 7));
    assert_ne!(mesh.sample(spacing, 7), mesh.sample(spacing, 8));
}

#[test]
fn spacing_larger_than_the_domain_samples_one_node() {
    assert_eq!(square(4).sample(Quantity::new(10.0), 1).len(), 1);
}

#[test]
#[should_panic(expected = "Sampling spacing must be positive.")]
fn nonpositive_spacing() {
    square(2).sample(Quantity::new(0.0), 1);
}
