use super::{Curve, arrange, junctions, meeting};
use crate::{
    geometry::Coordinate,
    math::{Tensor, assert::Assert},
};
use std::collections::{BTreeMap, HashMap, HashSet};

type Triangle = [Coordinate<3>; 3];

fn triangle(corners: [[f64; 3]; 3]) -> Triangle {
    corners.map(Coordinate::from)
}

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

#[test]
fn points_on_a_curve_are_put_in_order_from_its_start() {
    let at = |y: f64| triangle([[-1.0, y, -1.0], [-1.0, y, 3.0], [3.0, y, -1.0]]);
    let triangles = [
        triangle([[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]]),
        triangle([[0.0, -1.0, -1.0], [0.0, 3.0, -1.0], [0.0, -1.0, 3.0]]),
        at(0.2),
        at(0.6),
    ];
    let solids = [vec![0], vec![1], vec![2, 3]];
    let corners = |solid: usize, index: usize| {
        let triangle = &triangles[solids[solid][index]];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let mut pairs = BTreeMap::new();
    pairs.insert([0, 1], HashSet::from([[0, 0]]));
    pairs.insert([0, 2], HashSet::from([[0, 0], [0, 1]]));
    pairs.insert([1, 2], HashSet::from([[0, 0], [0, 1]]));
    let found = junctions(&pairs, 3, corners).unwrap();
    assert_eq!(found.len(), 2);
    let (low, high) = (
        Coordinate::from([0.0, -1.0, 0.0]),
        Coordinate::from([0.0, 2.0, 0.0]),
    );
    for reversed in [false, true] {
        let ends = if reversed {
            [high.clone(), low.clone()]
        } else {
            [low.clone(), high.clone()]
        };
        let curves: HashMap<Curve, [Coordinate<3>; 2]> = [
            (([0, 1], 0, 0), ends.clone()),
            (([0, 2], 0, 0), ends.clone()),
            (([0, 2], 0, 1), ends.clone()),
            (([1, 2], 0, 0), ends.clone()),
            (([1, 2], 0, 1), ends.clone()),
        ]
        .into();
        let (points, arranged) = arrange(&found, &curves, corners).unwrap();
        assert_eq!(points.len(), 2);
        let heights: Vec<f64> = arranged[&([0, 1], 0, 0)]
            .iter()
            .map(|&index| points[index].position[1].value())
            .collect();
        assert_eq!(heights.len(), 2);
        assert_eq!(heights[0] < heights[1], !reversed, "{heights:?}");
    }
}

#[test]
fn four_surfaces_through_one_point_are_one_point() {
    let triangles = [
        triangle([[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]]),
        triangle([[0.0, -1.0, -1.0], [0.0, 3.0, -1.0], [0.0, -1.0, 3.0]]),
        triangle([[-1.0, 0.0, -1.0], [-1.0, 0.0, 3.0], [3.0, 0.0, -1.0]]),
        triangle([[-1.0, -1.0, 2.0], [2.0, -1.0, -1.0], [-1.0, 2.0, -1.0]]),
    ];
    let corners = |solid: usize, _: usize| {
        let triangle = &triangles[solid];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let mut pairs = BTreeMap::new();
    for i in 0..4 {
        for j in i + 1..4 {
            pairs.insert([i, j], HashSet::from([[0, 0]]));
        }
    }
    let found = junctions(&pairs, 4, corners).unwrap();
    assert_eq!(found.len(), 4);
    let curves: HashMap<Curve, [Coordinate<3>; 2]> = pairs
        .keys()
        .map(|&pair| {
            (
                (pair, 0, 0),
                [Coordinate::from([0.0; 3]), Coordinate::from([1.0; 3])],
            )
        })
        .collect();
    let (points, arranged) = arrange(&found, &curves, corners).unwrap();
    assert_eq!(points.len(), 1);
    assert_eq!(points[0].triangles, [(0, 0), (1, 0), (2, 0), (3, 0)]);
    assert_eq!(arranged.len(), 6);
    assert!(arranged.values().all(|on| on == &[0]));
}
