use super::{
    Curve, Ends, Feature, Located, arrange, edge_junctions, junctions, meeting, planar_junctions,
};
use crate::{
    geometry::Coordinate,
    math::{FxHashMap, FxHashSet, Tensor, assert::Assert},
};
use alloc::collections::BTreeMap;
use core::array::from_fn;

type Triangle = [Coordinate<3>; 3];

fn away(points: [Coordinate<3>; 2]) -> Ends {
    Ends {
        points,
        features: from_fn(|_| from_fn(|_| Feature::Vertex(usize::MAX))),
    }
}

fn named(solid: usize, triangle: usize, located: Located) -> Feature {
    let triangle = 1000 * solid + triangle;
    match located {
        Located::Face => Feature::Face(triangle),
        Located::Edge(edge) => Feature::Edge([triangle, edge]),
        Located::Vertex(corner) => Feature::Vertex(10 * triangle + corner),
    }
}

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
    let (located, found) = meeting(borrowed(&triangles)).unwrap().unwrap();
    assert_eq!(located, [Located::Face; 3]);
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
fn a_point_on_the_edge_of_a_triangle_is_found_there() {
    let mut triangles = around([0.1, 0.2, 0.3]);
    triangles[1] = triangle([[0.0, 0.2, -1.0], [0.0, 2.0, -1.0], [0.0, 0.2, 2.0]]);
    let (located, _) = meeting(borrowed(&triangles)).unwrap().unwrap();
    assert_eq!(located, [Located::Face, Located::Edge(2), Located::Face]);
}

#[test]
fn a_point_on_the_corner_of_a_triangle_is_found_there() {
    let mut triangles = around([0.1, 0.2, 0.3]);
    triangles[1] = triangle([[0.0, 0.2, 0.3], [0.0, 2.0, -1.0], [0.0, -1.0, 2.0]]);
    let (located, _) = meeting(borrowed(&triangles)).unwrap().unwrap();
    assert_eq!(located, [Located::Face, Located::Vertex(0), Located::Face]);
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
        pairs.insert(pair, FxHashSet::from_iter([[0, 0]]));
    }
    let found = junctions(&pairs, 3, corners).unwrap();
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].triangles, [(0, 0), (1, 0), (2, 0)]);
    pairs.remove(&[1, 2]);
    assert!(junctions(&pairs, 3, corners).unwrap().is_empty());
    pairs.insert([1, 2], FxHashSet::from_iter([[1, 0]]));
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
    pairs.insert([0, 1], FxHashSet::from_iter([[0, 0]]));
    pairs.insert([0, 2], FxHashSet::from_iter([[0, 0], [0, 1]]));
    pairs.insert([1, 2], FxHashSet::from_iter([[0, 0], [0, 1]]));
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
        let curves: FxHashMap<Curve, Ends> = [
            (([0, 1], 0, 0), away(ends.clone())),
            (([0, 2], 0, 0), away(ends.clone())),
            (([0, 2], 0, 1), away(ends.clone())),
            (([1, 2], 0, 0), away(ends.clone())),
            (([1, 2], 0, 1), away(ends.clone())),
        ]
        .into_iter()
        .collect();
        let (points, arranged, _) = arrange(&found, &curves, named, corners).unwrap();
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
            pairs.insert([i, j], FxHashSet::from_iter([[0, 0]]));
        }
    }
    let found = junctions(&pairs, 4, corners).unwrap();
    assert_eq!(found.len(), 4);
    let curves: FxHashMap<Curve, Ends> = pairs
        .keys()
        .map(|&pair| {
            let ends = [Coordinate::from([0.0; 3]), Coordinate::from([1.0; 3])];
            ((pair, 0, 0), away(ends))
        })
        .collect();
    let (points, arranged, _) = arrange(&found, &curves, named, corners).unwrap();
    assert_eq!(points.len(), 1);
    let solids: Vec<usize> = points[0].places.iter().map(|&(solid, ..)| solid).collect();
    assert_eq!(solids, [0, 1, 2, 3]);
    assert_eq!(arranged.len(), 6);
    assert!(arranged.values().all(|on| on == &[0]));
}

fn across_coplanar() -> [Triangle; 3] {
    [
        triangle([[0.5, -2.0, -1.0], [0.5, 5.0, -1.0], [0.5, -2.0, 2.0]]),
        triangle([[-4.0, -4.0, 0.0], [8.0, -4.0, 0.0], [-4.0, 8.0, 0.0]]),
        triangle([[-1.0, -1.0, 0.0], [3.0, -1.0, 0.0], [-1.0, 3.0, 0.0]]),
    ]
}

#[test]
fn the_edges_of_a_triangle_in_the_plane_of_another_meet_a_third_where_it_crosses_them() {
    let triangles = across_coplanar();
    let corners = |solid: usize, _: usize| {
        let triangle = &triangles[solid];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let mut pairs = BTreeMap::new();
    pairs.insert([0, 1], FxHashSet::from_iter([[0, 0]]));
    pairs.insert([0, 2], FxHashSet::from_iter([[0, 0]]));
    let coplanar = BTreeMap::from([([1, 2], vec![[0, 0]])]);
    assert!(junctions(&pairs, 3, corners).unwrap().is_empty());
    let found = edge_junctions(&coplanar, &pairs, 3, corners).unwrap();
    assert_eq!(found.len(), 2);
    let mut heights: Vec<(f64, Vec<Located>)> = found
        .iter()
        .map(|junction| (junction.position[1].value(), junction.located.to_vec()))
        .collect();
    heights.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert!((heights[0].0 + 1.0).abs() < 1e-12 && (heights[1].0 - 1.5).abs() < 1e-12);
    assert_eq!(
        heights[0].1,
        [Located::Face, Located::Face, Located::Edge(0)]
    );
    assert_eq!(
        heights[1].1,
        [Located::Face, Located::Face, Located::Edge(1)]
    );
}

#[test]
fn an_edge_that_meets_the_plane_outside_the_other_triangle_is_not_a_junction() {
    let mut triangles = across_coplanar();
    triangles[0] = triangle([[0.5, 2.0, -1.0], [0.5, 5.0, -1.0], [0.5, 2.0, 2.0]]);
    let corners = |solid: usize, _: usize| {
        let triangle = &triangles[solid];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let mut pairs = BTreeMap::new();
    pairs.insert([0, 1], FxHashSet::from_iter([[0, 0]]));
    pairs.insert([0, 2], FxHashSet::from_iter([[0, 0]]));
    let coplanar = BTreeMap::from([([1, 2], vec![[0, 0]])]);
    assert!(
        edge_junctions(&coplanar, &pairs, 3, corners)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn points_defined_by_edges_are_put_in_order_along_a_curve() {
    let triangles = across_coplanar();
    let corners = |solid: usize, _: usize| {
        let triangle = &triangles[solid];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let mut pairs = BTreeMap::new();
    pairs.insert([0, 1], FxHashSet::from_iter([[0, 0]]));
    pairs.insert([0, 2], FxHashSet::from_iter([[0, 0]]));
    let coplanar = BTreeMap::from([([1, 2], vec![[0, 0]])]);
    let found = edge_junctions(&coplanar, &pairs, 3, corners).unwrap();
    for reversed in [false, true] {
        let ends = [
            Coordinate::from([0.5, -2.0, 0.0]),
            Coordinate::from([0.5, 8.0 / 3.0, 0.0]),
        ];
        let ends = if reversed {
            [ends[1].clone(), ends[0].clone()]
        } else {
            ends
        };
        let curves: FxHashMap<Curve, Ends> = [
            (([0, 1], 0, 0), away(ends.clone())),
            (([0, 2], 0, 0), away(ends)),
        ]
        .into_iter()
        .collect();
        let (points, arranged, _) = arrange(&found, &curves, named, corners).unwrap();
        assert_eq!(points.len(), 2);
        let heights: Vec<f64> = arranged[&([0, 1], 0, 0)]
            .iter()
            .map(|&index| points[index].position[1].value())
            .collect();
        assert_eq!(heights.len(), 2);
        assert_eq!(heights[0] < heights[1], !reversed, "{heights:?}");
    }
}

fn in_one_plane() -> [Triangle; 4] {
    [
        triangle([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 4.0, 0.0]]),
        triangle([[1.0, 2.0, 0.0], [5.0, 2.0, 0.0], [3.0, 5.0, 0.0]]),
        triangle([[-6.0, -6.0, 0.0], [12.0, -6.0, 0.0], [-6.0, 12.0, 0.0]]),
        triangle([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]),
    ]
}

#[test]
fn edges_that_cross_in_a_plane_three_triangles_share_make_junctions() {
    let triangles = in_one_plane();
    let list = [vec![0, 3], vec![1], vec![2]];
    let corners = |solid: usize, index: usize| {
        let triangle = &triangles[list[solid][index]];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let coplanar = BTreeMap::from([
        ([0, 1], vec![[0, 0]]),
        ([0, 2], vec![[0, 0]]),
        ([1, 2], vec![[0, 0]]),
    ]);
    let found = planar_junctions(&coplanar, &[2, 1, 1], corners).unwrap();
    let mut places: Vec<(f64, f64, Vec<Located>)> = found
        .iter()
        .map(|junction| {
            (
                junction.position[0].value(),
                junction.position[1].value(),
                junction.located.to_vec(),
            )
        })
        .collect();
    places.sort_by(|a, b| a.0.total_cmp(&b.0));
    assert_eq!(places.len(), 2);
    assert!((places[0].0 - 1.4).abs() < 1e-12 && (places[0].1 - 2.6).abs() < 1e-12);
    assert!((places[1].0 - 2.0).abs() < 1e-12 && (places[1].1 - 2.0).abs() < 1e-12);
    assert_eq!(
        places[0].2,
        [Located::Edge(1), Located::Edge(2), Located::Face]
    );
    assert_eq!(
        places[1].2,
        [Located::Edge(1), Located::Edge(0), Located::Face]
    );
}

#[test]
fn edges_that_cross_outside_the_third_triangle_make_no_junction() {
    let mut triangles = in_one_plane();
    triangles[2] = triangle([[-6.0, -6.0, 0.0], [1.0, -6.0, 0.0], [-6.0, 1.0, 0.0]]);
    let list = [vec![0, 3], vec![1], vec![2]];
    let corners = |solid: usize, index: usize| {
        let triangle = &triangles[list[solid][index]];
        [&triangle[0], &triangle[1], &triangle[2]]
    };
    let coplanar = BTreeMap::from([
        ([0, 1], vec![[0, 0]]),
        ([0, 2], vec![[0, 0]]),
        ([1, 2], vec![[0, 0]]),
    ]);
    assert!(
        planar_junctions(&coplanar, &[2, 1, 1], corners)
            .unwrap()
            .is_empty()
    );
}
