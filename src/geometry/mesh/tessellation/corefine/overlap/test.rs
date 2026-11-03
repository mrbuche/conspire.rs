use super::{Constraint, Side};
use crate::{
    geometry::{
        Coordinate,
        bvh::closest_point_on_triangle,
        mesh::{
            Connectivity, Mesh, Tessellation,
            tessellation::corefine::{Feature, Key},
            test::cuboid,
        },
    },
    math::Tensor,
    units::Length,
};
use std::array::from_fn;

fn square(corner: [f64; 2], size: f64, triangles: [[usize; 3]; 2]) -> Tessellation {
    let [x, y] = corner;
    let coordinates: Vec<Coordinate<3>> = [
        [x, y, 0.0],
        [x + size, y, 0.0],
        [x + size, y + size, 0.0],
        [x, y + size, 0.0],
    ]
    .map(Into::into)
    .to_vec();
    Mesh::from((
        vec![Connectivity::Triangular(triangles.to_vec().into())],
        coordinates.into(),
    ))
    .into()
}

const RISING: [[usize; 3]; 2] = [[0, 1, 2], [0, 2, 3]];
const FALLING: [[usize; 3]; 2] = [[0, 1, 3], [1, 2, 3]];

fn vertex(node: usize) -> Feature {
    Feature::Vertex(node)
}

#[test]
fn squares_with_crossing_diagonals_split_into_four_triangles() {
    let rising = square([0.0, 0.0], 1.0, RISING);
    let falling = square([0.0, 0.0], 1.0, FALLING);
    let constraints = rising.coplanar_constraints(&falling).unwrap();
    assert_eq!(constraints.len(), 4);
    let center = Key {
        first: Feature::Edge([0, 2]),
        second: Feature::Edge([1, 3]),
    };
    let mut corners: Vec<(Side, usize, Key)> = constraints
        .iter()
        .map(|constraint| {
            let [a, b] = &constraint.keys;
            let corner = if *a == center { b } else { a };
            assert!(*a == center || *b == center);
            (constraint.side, constraint.triangle, corner.clone())
        })
        .collect();
    corners.sort_by_key(|(side, triangle, _)| (*side == Side::Second, *triangle));
    let expected = [
        (Side::First, 0, 1),
        (Side::First, 1, 3),
        (Side::Second, 0, 0),
        (Side::Second, 1, 2),
    ];
    corners.iter().zip(expected).for_each(
        |((side, triangle, key), (expected_side, expected_triangle, node))| {
            assert_eq!((*side, *triangle), (expected_side, expected_triangle));
            assert_eq!(
                *key,
                Key {
                    first: vertex(node),
                    second: vertex(node)
                }
            );
        },
    );
    constraints.iter().for_each(|constraint| {
        let ends: Vec<[f64; 2]> = constraint
            .segment
            .iter()
            .map(|point| [point[0].value(), point[1].value()])
            .collect();
        assert!(
            ends.iter()
                .any(|end| (end[0] - 0.5).abs() < 1e-12 && (end[1] - 0.5).abs() < 1e-12)
        );
    });
}

#[test]
fn identical_triangulations_need_no_constraints() {
    let first = square([0.0, 0.0], 1.0, RISING);
    let second = square([0.0, 0.0], 1.0, RISING);
    assert!(!first.coplanar_pairs(&second).unwrap().is_empty());
    assert!(first.coplanar_constraints(&second).unwrap().is_empty());
}

#[test]
fn cubes_sharing_a_face_with_matching_diagonals_need_no_constraints() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([1.0, 0.0, 0.0], [2.0, 1.0, 1.0]);
    assert!(first.coplanar_constraints(&second).unwrap().is_empty());
}

fn distance_to_segment(point: [f64; 3], segment: [[f64; 3]; 2]) -> f64 {
    let [a, b] = segment;
    let direction: [f64; 3] = from_fn(|i| b[i] - a[i]);
    let length_squared: f64 = direction.iter().map(|x| x * x).sum();
    let along: f64 = (0..3)
        .map(|i| (point[i] - a[i]) * direction[i])
        .sum::<f64>()
        / length_squared;
    let closest: [f64; 3] = std::array::from_fn(|i| a[i] + along.clamp(0.0, 1.0) * direction[i]);
    (0..3)
        .map(|i| (point[i] - closest[i]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn check_geometry(first: &Tessellation, second: &Tessellation, constraints: &[Constraint]) {
    constraints.iter().for_each(|constraint| {
        let (split, other) = match constraint.side {
            Side::First => (first, second),
            Side::Second => (second, first),
        };
        let triangle: Vec<&Coordinate<3>> = split
            .mesh()
            .connectivities()
            .iter()
            .flatten()
            .nth(constraint.triangle)
            .unwrap()
            .iter()
            .map(|&node| &split.mesh().coordinates()[node])
            .collect();
        constraint.segment.iter().for_each(|end| {
            let nearest = closest_point_on_triangle(end, triangle[0], triangle[1], triangle[2]);
            assert!((&nearest - end).norm() < Length::meters(1e-12));
        });
        let ends = constraint
            .segment
            .each_ref()
            .map(|end| [0, 1, 2].map(|i| end[i].value()));
        let on_an_edge = other.mesh().connectivities().iter().flatten().any(|nodes| {
            (0..3).any(|edge| {
                let edge = [nodes[edge], nodes[(edge + 1) % 3]]
                    .map(|node| [0, 1, 2].map(|i| other.mesh().coordinates()[node][i].value()));
                ends.iter()
                    .all(|&end| distance_to_segment(end, edge) < 1e-12)
            })
        });
        assert!(on_an_edge, "{:?}", constraint.keys);
    })
}

#[test]
fn a_partial_overlap_is_cut_along_pieces_of_the_other_edges() {
    let first = square([0.0, 0.0], 1.0, RISING);
    let second = square([0.3, 0.45], 1.0, FALLING);
    let constraints = first.coplanar_constraints(&second).unwrap();
    assert!(!constraints.is_empty());
    check_geometry(&first, &second, &constraints);
    assert!(
        constraints
            .iter()
            .any(|constraint| constraint.side == Side::First)
    );
    assert!(
        constraints
            .iter()
            .any(|constraint| constraint.side == Side::Second)
    );
}
