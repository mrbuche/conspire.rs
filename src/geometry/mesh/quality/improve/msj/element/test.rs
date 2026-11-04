use super::super::{
    kind::{HEXAHEDRON_POINTS, KINDS},
    test::{Raw, UNIT, distorted, offset, terms, typed},
};
use super::Frame;
use super::bounded;
use crate::{
    geometry::CoordinateList,
    math::{
        Quantity, Scalar, Tensor,
        assert::{Assert, AssertionError},
    },
};
use std::{array::from_fn, f64::consts::SQRT_2};

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
        HEXAHEDRON_POINTS
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

const WEDGE_KIND: usize = 2;

const PYRAMID_KIND: usize = 3;

fn unit_wedge() -> Vec<[Scalar; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
}

fn unit_pyramid() -> Vec<[Scalar; 3]> {
    vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.6],
    ]
}

fn shaken(nodes: &[[Scalar; 3]], scale: Scalar, seed: Scalar) -> Vec<[Scalar; 3]> {
    let mut nodes = nodes.to_vec();
    nodes
        .iter_mut()
        .enumerate()
        .for_each(|(n, node)| (0..3).for_each(|i| node[i] += scale * offset(3 * n + i, seed)));
    nodes
}

fn padded(nodes: &[[Scalar; 3]]) -> CoordinateList<3, 8> {
    typed(&from_fn(|i| nodes[if i < nodes.len() { i } else { 0 }]))
}

fn kind_value(kind: usize, theta: Scalar, scale: Scalar, nodes: &[[Scalar; 3]]) -> Scalar {
    bounded(
        &KINDS[kind],
        Quantity::new(theta),
        Quantity::new(scale),
        &padded(nodes),
        false,
    )
    .0
    .value()
}

fn mean_edge_of(kind: usize, nodes: &[[Scalar; 3]]) -> Scalar {
    let edges = KINDS[kind].edges;
    edges
        .iter()
        .map(|&[a, b]| {
            (0..3)
                .map(|i| (nodes[a][i] - nodes[b][i]).powi(2))
                .sum::<Scalar>()
                .sqrt()
        })
        .sum::<Scalar>()
        / edges.len() as Scalar
}

fn kind_check(kind: usize, theta: Scalar, nodes: &[[Scalar; 3]]) {
    let (_, analytic, _) = bounded(
        &KINDS[kind],
        Quantity::new(theta),
        Quantity::new(1.3),
        &padded(nodes),
        true,
    );
    let step = 1.0e-6;
    (0..8).for_each(|n| {
        (0..3).for_each(|i| {
            let numerical = if n < nodes.len() {
                let (mut above, mut below) = (nodes.to_vec(), nodes.to_vec());
                above[n][i] += step;
                below[n][i] -= step;
                (kind_value(kind, theta, 1.3, &above) - kind_value(kind, theta, 1.3, &below))
                    / (2.0 * step)
            } else {
                0.0
            };
            let exact = analytic[n][i].value();
            assert!(
                (exact - numerical).abs() < 1.0e-6,
                "kind {kind} node {n} axis {i}: {exact} vs {numerical}"
            );
        })
    })
}

#[test]
fn wedge_and_pyramid_gradients_match_finite_difference_on_both_branches() {
    kind_check(WEDGE_KIND, 2.0, &shaken(&unit_wedge(), 0.15, 1.0));
    let mut wedge = shaken(&unit_wedge(), 0.05, 2.0);
    wedge[3][2] = -0.6;
    kind_check(WEDGE_KIND, 0.5, &wedge);
    kind_check(PYRAMID_KIND, 2.0, &shaken(&unit_pyramid(), 0.1, 3.0));
    let mut pyramid = shaken(&unit_pyramid(), 0.05, 4.0);
    pyramid[4][2] = -0.4;
    kind_check(PYRAMID_KIND, 0.5, &pyramid);
}

#[test]
fn wedges_and_pyramids_score_the_scaled_jacobian_of_the_metric() -> Result<(), AssertionError> {
    use crate::geometry::mesh::quality::metrics::{pyramid, wedge};
    let determinant = |e: &[[Scalar; 3]; 3]| {
        e[0][0] * (e[1][1] * e[2][2] - e[1][2] * e[2][1])
            - e[0][1] * (e[1][0] * e[2][2] - e[1][2] * e[2][0])
            + e[0][2] * (e[1][0] * e[2][1] - e[1][1] * e[2][0])
    };
    let norm = |v: &[Scalar; 3]| v.iter().map(|x| x * x).sum::<Scalar>().sqrt();
    let reference = |table: &[(usize, [usize; 3])], c: Scalar, nodes: &[[Scalar; 3]]| {
        table
            .iter()
            .map(|(corner, adjacent)| {
                let e: [[Scalar; 3]; 3] =
                    from_fn(|k| from_fn(|i| nodes[adjacent[k]][i] - nodes[*corner][i]));
                c * determinant(&e) / (norm(&e[0]) * norm(&e[1]) * norm(&e[2]))
            })
            .fold(Scalar::INFINITY, Scalar::min)
    };
    for seed in 0..8 {
        for (kind, table, c, base, noise) in [
            (
                WEDGE_KIND,
                &wedge::CORNERS[..],
                2.0 / 3.0_f64.sqrt(),
                unit_wedge(),
                0.15,
            ),
            (
                PYRAMID_KIND,
                &pyramid::CORNERS[..],
                SQRT_2,
                unit_pyramid(),
                0.1,
            ),
        ] {
            let nodes = shaken(&base, noise, seed as Scalar);
            let expected = reference(table, c, &nodes);
            assert!(expected > 0.0);
            let scale = mean_edge_of(kind, &nodes);
            let theta = 2.0;
            Assert::default().eq_within_tols(
                kind_value(kind, theta, scale, &nodes) / (scale * scale) + theta,
                expected,
            )?
        }
    }
    Ok(())
}

#[test]
fn a_regular_wedge_and_a_regular_pyramid_score_one() -> Result<(), AssertionError> {
    let wedge = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 0.75_f64.sqrt(), 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.5, 0.75_f64.sqrt(), 1.0],
    ];
    let mut pyramid = unit_pyramid();
    pyramid[4][2] = 0.5_f64.sqrt();
    for (kind, nodes) in [(WEDGE_KIND, wedge), (PYRAMID_KIND, pyramid)] {
        let scale = mean_edge_of(kind, &nodes);
        Assert::default().eq_within_tols(
            kind_value(kind, 2.0, scale, &nodes) / (scale * scale) + 2.0,
            1.0,
        )?
    }
    Ok(())
}

#[test]
fn wedge_and_pyramid_energies_are_continuous_through_a_flat_element() -> Result<(), AssertionError>
{
    let delta = 1.0e-7;
    let theta = 0.3;
    let wedge = |t: Scalar| {
        let mut nodes = unit_wedge();
        nodes[3] = [0.6, 0.0, t];
        nodes
    };
    let pyramid = |h: Scalar| {
        let mut nodes = unit_pyramid();
        nodes[4] = [0.3, 0.4, h];
        nodes
    };
    for (kind, shape) in [
        (WEDGE_KIND, &wedge as &dyn Fn(Scalar) -> Vec<[Scalar; 3]>),
        (PYRAMID_KIND, &pyramid),
    ] {
        let energy = |t: Scalar| {
            let nodes = shape(t);
            kind_value(kind, theta, mean_edge_of(kind, &nodes), &nodes)
        };
        let (right, left) = (
            (energy(2.0 * delta) - energy(delta)) / delta,
            (energy(-delta) - energy(-2.0 * delta)) / delta,
        );
        Assert {
            abs_tol: 1.0e-5,
            rel_tol: 1.0e-5,
            ..Default::default()
        }
        .eq_within_tols(energy(delta), energy(-delta))?;
        assert!(right.is_finite() && left.is_finite());
        assert!(right * left > 0.0 && (right / left).abs() < 3.0 && (left / right).abs() < 3.0);
    }
    Ok(())
}
