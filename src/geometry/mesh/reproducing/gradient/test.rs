use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{
            Basis, Connectivity, Mesh,
            test::{square, tetrahedra},
        },
    },
    math::{FxHashSet, Quantity},
    units::Length,
};
use std::array::from_fn;

const TOLERANCE: f64 = 1e-9;

fn length(value: f64) -> Quantity<Length> {
    Quantity::new(value)
}

fn point<const D: usize>(mesh: &Mesh<D>, node: usize) -> [f64; D] {
    from_fn(|k| mesh.coordinates()[node][k].value())
}

fn bases<const D: usize>(
    mesh: &Mesh<D>,
    approximation: (f64, f64),
    quadrature: (f64, f64),
) -> (Basis, Basis) {
    let build = |(h, reach): (f64, f64), seed: u64| {
        let seeds = mesh.sample(length(h), seed);
        mesh.reproducing_basis(&seeds, length(reach * h), 1)
            .unwrap()
    };
    (build(approximation, 3), build(quadrature, 5))
}

fn assert_consistent<const D: usize>(mesh: &Mesh<D>, approximation: &Basis, quadrature: &Basis) {
    let gradients = mesh.projected_gradients(approximation, quadrature).unwrap();
    assert_eq!(gradients.values.len(), quadrature.seeds.len());
    let slope: [f64; D] = from_fn(|k| 3.0 - 2.5 * k as f64);
    let linear = |x: [f64; D]| 1.0 + (0..D).map(|k| slope[k] * x[k]).sum::<f64>();
    for entries in &gradients.values {
        assert!(!entries.is_empty());
        assert!(entries.windows(2).all(|w| w[0].0 < w[1].0));
        for k in 0..D {
            let sum: f64 = entries.iter().map(|(_, g)| g[k].value()).sum();
            assert!(sum.abs() < TOLERANCE, "gradients do not sum to zero: {sum}");
            let reproduced: f64 = entries
                .iter()
                .map(|&(function, ref g)| {
                    linear(point(mesh, approximation.seeds[function])) * g[k].value()
                })
                .sum();
            assert!(
                (reproduced - slope[k]).abs() < TOLERANCE,
                "linear field, axis {k}: {reproduced} vs {}",
                slope[k]
            );
        }
    }
}

#[test]
fn consistent_in_two_dimensions() {
    let mesh = square(40);
    let (approximation, quadrature) = bases(&mesh, (0.3, 2.6), (0.075, 2.6));
    assert_consistent(&mesh, &approximation, &quadrature);
}

#[test]
fn integrate_to_the_gradient_of_each_function() {
    let mesh = square(40);
    let (approximation, quadrature) = bases(&mesh, (0.3, 2.6), (0.075, 2.6));
    let gradients = mesh
        .projected_gradients(&approximation, &quadrature)
        .unwrap();
    let weights = mesh.integrals(&quadrature).unwrap();
    let mut projected = vec![[0.0; 2]; approximation.seeds.len()];
    for (entries, weight) in gradients.values.iter().zip(&weights) {
        for (function, g) in entries {
            (0..2).for_each(|k| projected[*function][k] += weight.value() * g[k].value());
        }
    }
    let elements: Vec<usize> = (0..mesh.number_of_elements()).collect();
    let simplices = mesh.simplices_over::<3>(&elements).unwrap();
    let mut direct = vec![[0.0; 2]; approximation.seeds.len()];
    for simplex in &simplices {
        let mut gradient = vec![[0.0; 2]; approximation.seeds.len()];
        for (function, values) in approximation.values.iter().enumerate() {
            for &(node, value) in values {
                if let Some(a) = simplex.nodes.iter().position(|&n| n == node) {
                    (0..2).for_each(|k| gradient[function][k] += value * simplex.gradients[a][k]);
                }
            }
        }
        for (function, g) in gradient.iter().enumerate() {
            (0..2).for_each(|k| direct[function][k] += simplex.volume * g[k]);
        }
    }
    for (a, b) in projected.iter().zip(&direct) {
        (0..2).for_each(|k| assert!((a[k] - b[k]).abs() < TOLERANCE, "{a:?} vs {b:?}"));
    }
}

#[test]
fn consistent_in_three_dimensions() {
    let mesh = tetrahedra(8);
    let (approximation, quadrature) = bases(&mesh, (0.5, 2.2), (0.25, 3.6));
    assert_consistent(&mesh, &approximation, &quadrature);
}

#[test]
fn each_quadrature_function_sees_only_nearby_functions() {
    let mesh = square(40);
    let (approximation, quadrature) = bases(&mesh, (0.15, 2.6), (0.075, 2.6));
    let gradients = mesh
        .projected_gradients(&approximation, &quadrature)
        .unwrap();
    let elements = |basis: &Basis, function: usize| -> FxHashSet<usize> {
        basis.values[function]
            .iter()
            .flat_map(|&(node, _)| mesh.node_element_connectivity()[node].iter().copied())
            .collect()
    };
    let mut disjoint = 0;
    for (point, entries) in gradients.values.iter().enumerate() {
        let support = elements(&quadrature, point);
        for function in 0..approximation.seeds.len() {
            if elements(&approximation, function).is_disjoint(&support) {
                disjoint += 1;
                assert!(
                    entries.iter().all(|&(f, _)| f != function),
                    "a function with a disjoint support has a gradient"
                );
            }
        }
    }
    assert!(disjoint > 0, "every pair of supports overlapped");
}

#[test]
fn errors() {
    let mesh = square(4);
    let empty = Basis {
        seeds: vec![0],
        values: vec![vec![(0, 0.0)]],
    };
    let basis = Basis {
        seeds: vec![0],
        values: vec![vec![(0, 1.0)]],
    };
    assert_eq!(
        mesh.projected_gradients(&basis, &empty).unwrap_err(),
        "a quadrature function has no positive integral"
    );
    let quadrilateral = Mesh::from((
        vec![Connectivity::Quadrilateral(vec![[0usize, 1, 2, 3]].into())],
        Coordinates::from([
            Coordinate::from([0.0, 0.0]),
            Coordinate::from([1.0, 0.0]),
            Coordinate::from([1.0, 1.0]),
            Coordinate::from([0.0, 1.0]),
        ]),
    ));
    assert_eq!(
        quadrilateral
            .projected_gradients(&basis, &basis)
            .unwrap_err(),
        "projected gradients require a triangular or tetrahedral mesh"
    );
}
