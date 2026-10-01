use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Connectivity, Mesh,
            test::{square, tetrahedra},
        },
    },
    math::Quantity,
    units::Length,
};
use std::{array::from_fn, f64::consts::PI};

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point<const D: usize>(mesh: &Mesh<D>, node: usize) -> [f64; D] {
    from_fn(|k| mesh.coordinates()[node][k].value())
}

fn quadrature_error(mesh: &Mesh<2>, h: f64, seed: u64) -> f64 {
    let seeds = mesh.sample(length(h), seed);
    let basis = mesh
        .reproducing_basis(&seeds, length(2.6 * h), 1, 1)
        .unwrap();
    let weights = mesh.integrals(&basis).unwrap();
    let sum: f64 = seeds
        .iter()
        .zip(&weights)
        .map(|(&node, w)| {
            let [x, y] = point(mesh, node);
            w.value() * (PI * x).sin() * (PI * y).sin()
        })
        .sum();
    (sum - 4.0 / (PI * PI)).abs()
}

#[test]
fn weights_integrate_constants_and_linear_fields() {
    let mesh = square(40);
    let seeds = mesh.sample(length(0.2), 3);
    let basis = mesh
        .reproducing_basis(&seeds, length(2.6 * 0.2), 1, 1)
        .unwrap();
    let weights: Vec<f64> = mesh
        .integrals(&basis)
        .unwrap()
        .iter()
        .map(|w| w.value())
        .collect();
    assert!(weights.iter().all(|&w| w > 0.0), "a weight is not positive");
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for k in 0..2 {
        let first: f64 = seeds
            .iter()
            .zip(&weights)
            .map(|(&node, w)| w * point(&mesh, node)[k])
            .sum();
        assert!(
            (first - 0.5).abs() < 1e-9,
            "linear field, axis {k}: {first}"
        );
    }
}

#[test]
fn weights_in_three_dimensions() {
    let mesh = tetrahedra(6);
    let seeds = mesh.sample(length(0.4), 2);
    let basis = mesh.reproducing_basis(&seeds, length(1.1), 1, 1).unwrap();
    let weights: Vec<f64> = mesh
        .integrals(&basis)
        .unwrap()
        .iter()
        .map(|w| w.value())
        .collect();
    assert!((weights.iter().sum::<f64>() - 1.0).abs() < 1e-9);
    for k in 0..3 {
        let first: f64 = seeds
            .iter()
            .zip(&weights)
            .map(|(&node, w)| w * point(&mesh, node)[k])
            .sum();
        assert!(
            (first - 0.5).abs() < 1e-9,
            "linear field, axis {k}: {first}"
        );
    }
}

#[test]
fn quadrature_converges_at_second_order() {
    let mesh = square(40);
    let errors: Vec<f64> = [0.4, 0.2, 0.1]
        .iter()
        .map(|&h| quadrature_error(&mesh, h, 1))
        .collect();
    assert!(errors[0] / errors[1] > 2.0, "{errors:?}");
    assert!(errors[1] / errors[2] > 2.0, "{errors:?}");
}

#[test]
fn only_simplicial_meshes_have_weights() {
    let quadrilateral = Mesh::from((
        vec![Connectivity::Quadrilateral(vec![[0usize, 1, 2, 3]].into())],
        Coordinates::from([
            Coordinate::from([0.0, 0.0]),
            Coordinate::from([1.0, 0.0]),
            Coordinate::from([1.0, 1.0]),
            Coordinate::from([0.0, 1.0]),
        ]),
    ));
    let basis = crate::geometry::mesh::Basis {
        seeds: vec![0],
        values: vec![vec![(0, 1.0)]],
    };
    assert_eq!(
        quadrilateral.integrals(&basis).unwrap_err(),
        "quadrature weights require a triangular or tetrahedral mesh"
    );
}
