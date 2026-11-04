use super::{junctions, meeting};
use crate::{
    geometry::Coordinate,
    math::{Tensor, assert::Assert},
};
use std::collections::{BTreeMap, HashSet};

type Triangle = [Coordinate<3>; 3];

fn triangle(corners: [[f64; 3]; 3]) -> Triangle {
    corners.map(Coordinate::from)
}

/// Three triangles, one in each coordinate plane, around a point on the way to the origin.
fn around(origin: [f64; 3]) -> [Triangle; 3] {
    let [x, y, z] = origin;
    [
        triangle([
            [x - 1.0, y - 1.0, z],
            [x + 2.0, y - 1.0, z],
            [x - 1.0, y + 2.0, z],
        ]),
        triangle([
            [x, y - 1.0, z - 1.0],
            [x, y + 2.0, z - 1.0],
            [x, y - 1.0, z + 2.0],
        ]),
        triangle([
            [x - 1.0, y, z - 1.0],
            [x - 1.0, y, z + 2.0],
            [x + 2.0, y, z - 1.0],
        ]),
    ]
}

fn borrowed(triangles: &[Triangle; 3]) -> [[&Coordinate<3>; 3]; 3] {
    triangles.each_ref().map(|t| [&t[0], &t[1], &t[2]])
}

#[test]
fn three_triangles_meet_at_a_point_inside_all_of_them() {
    let triangles = around([0.1, 0.2, 0.3]);
    let found = meeting(borrowed(&triangles)).unwrap().unwrap();
    let expected = Coordinate::from([0.1, 0.2, 0.3]);
    Assert::default()
        .zero_within_tols(&(&found - &expected).norm())
        .unwrap();
}

#[test]
fn a_point_outside_one_triangle_is_not_a_junction() {
    let mut triangles = around([0.1, 0.2, 0.3]);
    triangles[1] = triangle([[0.0, 0.5, -1.0], [0.0, 2.0, -1.0], [0.0, 0.5, 2.0]]);
    assert!(meeting(borrowed(&triangles)).unwrap().is_none());
}

#[test]
fn a_point_on_the_edge_of_a_triangle_is_an_error() {
    let mut triangles = around([0.1, 0.2, 0.3]);
    triangles[1] = triangle([[0.0, 0.2, -1.0], [0.0, 2.0, -1.0], [0.0, 0.2, 2.0]]);
    assert!(meeting(borrowed(&triangles)).is_err());
}

#[test]
fn planes_through_one_line_are_an_error() {
    let triangles = [
        triangle([[0.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 2.0]]),
        triangle([[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 0.0, 2.0]]),
        triangle([[0.0, 0.0, -1.0], [1.0, 1.0, -1.0], [0.0, 0.0, 2.0]]),
    ];
    assert!(meeting(borrowed(&triangles)).is_err());
}

#[test]
fn junctions_come_from_triangles_that_cross_pairwise() {
    let solids = [
        around([0.1, 0.2, 0.3]).into_iter().next().unwrap(),
        around([0.1, 0.2, 0.3]).into_iter().nth(1).unwrap(),
        around([0.1, 0.2, 0.3]).into_iter().nth(2).unwrap(),
    ];
    let corners = |solid: usize, _: usize| {
        let triangle = &solids[solid];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let mut pairs = BTreeMap::new();
    for pair in [[0, 1], [0, 2], [1, 2]] {
        pairs.insert(pair, HashSet::from([[0, 0]]));
    }
    let found = junctions(&pairs, 3, corners).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].triangles, [(0, 0), (1, 0), (2, 0)]);
    pairs.remove(&[1, 2]);
    assert!(junctions(&pairs, 3, corners).unwrap().is_empty());
    pairs.insert([1, 2], HashSet::from([[1, 0]]));
    assert!(junctions(&pairs, 3, corners).unwrap().is_empty());
}
