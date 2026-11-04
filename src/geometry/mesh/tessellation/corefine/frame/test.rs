use super::{Frame, frame};
use crate::{
    geometry::{
        Coordinate,
        mesh::tessellation::corefine::{Feature, Key, Side, triangulate::triangulate},
        predicates::Cutter,
    },
    math::assert::{Assert, AssertionError},
};

type Plane = [Coordinate<3>; 3];

fn point(x: f64, y: f64, z: f64) -> Coordinate<3> {
    [x, y, z].into()
}

fn plane(axis: usize, c: f64) -> Plane {
    match axis {
        0 => [point(c, 0.0, 0.0), point(c, 1.0, 0.0), point(c, 0.0, 1.0)],
        _ => [point(0.0, c, 0.0), point(1.0, c, 0.0), point(0.0, c, 1.0)],
    }
}

const NODES: [usize; 3] = [10, 11, 12];

fn right_triangle() -> [Coordinate<3>; 3] {
    [
        point(0.0, 0.0, 0.0),
        point(64.0, 0.0, 0.0),
        point(0.0, 64.0, 0.0),
    ]
}

fn edge_key(edge: usize, plane: usize) -> Key {
    let (a, b) = (NODES[edge], NODES[(edge + 1) % 3]);
    Key {
        first: Feature::Edge([a.min(b), a.max(b)]),
        second: Feature::Face(plane),
    }
}

fn cutter<'a>(planes: &'a [Plane]) -> impl Fn(&Feature) -> Cutter<'a> + 'a {
    move |feature| match feature {
        Feature::Face(i) => Cutter::Face([&planes[*i][0], &planes[*i][1], &planes[*i][2]]),
        _ => panic!("only planes cut"),
    }
}

fn run(
    triangle: &[Coordinate<3>; 3],
    points: &[(Key, Coordinate<3>)],
    segments: &[[Key; 2]],
    planes: &[Plane],
) -> Result<Frame, &'static str> {
    frame(
        [&triangle[0], &triangle[1], &triangle[2]],
        NODES,
        Side::First,
        points,
        segments,
        cutter(planes),
    )
}

fn planes_along(found: &Frame, edge: usize) -> Vec<usize> {
    found.input.sides[edge]
        .iter()
        .map(|&index| match found.keys[index - 3].second {
            Feature::Face(i) => i,
            _ => unreachable!(),
        })
        .collect()
}

fn garbage() -> Coordinate<3> {
    point(1.0, 1.0, 0.0)
}

const OFFSETS: [f64; 6] = [3.0, 40.0, 7.0, 21.0, 50.0, 13.0];

#[test]
fn points_on_an_edge_are_ordered_by_what_cuts_them_not_by_their_coordinates() {
    let triangle = right_triangle();
    for edge in 0..3 {
        let axis = if edge == 2 { 1 } else { 0 };
        let planes: Vec<Plane> = OFFSETS.iter().map(|&c| plane(axis, c)).collect();
        let points: Vec<_> = (0..planes.len())
            .map(|i| (edge_key(edge, i), garbage()))
            .collect();
        let found = run(&triangle, &points, &[], &planes).unwrap();
        let order = planes_along(&found, edge);
        let mut sorted: Vec<usize> = (0..OFFSETS.len()).collect();
        sorted.sort_by(|&a, &b| OFFSETS[a].total_cmp(&OFFSETS[b]));
        if edge != 0 {
            sorted.reverse();
        }
        assert_eq!(order, sorted, "edge {edge}");
        let parameters: Vec<f64> = found.input.sides[edge]
            .iter()
            .map(|&index| {
                let [u, v] = found.input.points[index];
                match edge {
                    0 => u,
                    1 => v,
                    _ => -v,
                }
            })
            .collect();
        assert!(
            parameters.windows(2).all(|pair| pair[0] < pair[1]),
            "edge {edge}: {parameters:?}"
        );
    }
}

#[test]
fn points_are_exactly_on_the_edges_of_the_frame() {
    let triangle = right_triangle();
    for edge in 0..3 {
        let axis = if edge == 2 { 1 } else { 0 };
        let offsets: Vec<f64> = (1..40).map(|i| 1.0 / 3.0 * i as f64 + 1e-9).collect();
        let planes: Vec<Plane> = offsets.iter().map(|&c| plane(axis, c)).collect();
        let points: Vec<_> = (0..planes.len())
            .map(|i| {
                let c = offsets[i];
                let on = match edge {
                    0 => point(c, 0.0, 0.0),
                    1 => point(c, 64.0 - c, 0.0),
                    _ => point(0.0, c, 0.0),
                };
                (edge_key(edge, i), on)
            })
            .collect();
        let found = run(&triangle, &points, &[], &planes).unwrap();
        assert_eq!(found.input.sides[edge].len(), offsets.len());
        for &index in &found.input.sides[edge] {
            let [u, v] = found.input.points[index];
            match edge {
                0 => assert_eq!(v, 0.0),
                2 => assert_eq!(u, 0.0),
                _ => {
                    assert_eq!(u + v, 1.0);
                    assert_eq!(1.0 - u, v);
                    assert_eq!(v * (1u64 << 52) as f64, (v * (1u64 << 52) as f64).round());
                }
            }
        }
    }
}

#[test]
fn points_too_close_to_tell_apart_on_the_diagonal_edge_fail_but_not_elsewhere() {
    let triangle = right_triangle();
    let planes = vec![plane(0, 1e-17), plane(0, 2e-17)];
    let at = |edge: usize| -> Vec<(Key, Coordinate<3>)> {
        let x = [1e-17, 2e-17];
        (0..2)
            .map(|i| {
                let on = if edge == 0 {
                    point(x[i], 0.0, 0.0)
                } else {
                    point(x[i], 64.0, 0.0)
                };
                (edge_key(edge, i), on)
            })
            .collect()
    };
    assert!(run(&triangle, &at(0), &[], &planes).is_ok());
    assert!(run(&triangle, &at(1), &[], &planes).is_err());
}

#[test]
fn interior_points_are_placed_by_barycentric_coordinates_in_the_metric_of_the_plane() {
    let triangle = [
        point(0.0, 0.0, 0.0),
        point(2.0, 0.0, 1.0),
        point(0.0, 3.0, 1.0),
    ];
    let key = Key {
        first: Feature::Face(0),
        second: Feature::Face(0),
    };
    let found = run(
        &triangle,
        &[(key.clone(), point(1.0, 0.75, 0.75))],
        &[],
        &[],
    )
    .unwrap();
    assert_eq!(found.input.gram, [5.0, 1.0, 10.0]);
    assert_eq!(found.input.points[3], [0.5, 0.25]);
    assert_eq!(found.keys, vec![key]);
}

#[test]
fn interior_points_rounded_out_of_the_triangle_fail() {
    let triangle = right_triangle();
    let key = |i| Key {
        first: Feature::Face(0),
        second: Feature::Face(i),
    };
    for position in [
        point(70.0, 1.0, 0.0),
        point(-1.0, 1.0, 0.0),
        point(1.0, -1.0, 0.0),
        point(32.0, 32.1, 0.0),
        point(10.0, -1e-6, 0.0),
    ] {
        assert!(run(&triangle, &[(key(0), position)], &[], &[]).is_err());
    }
    assert!(run(&triangle, &[(key(0), point(10.0, 10.0, 0.0))], &[], &[]).is_ok());
}

#[test]
fn interior_points_rounded_onto_or_just_past_the_boundary_are_nudged_inside()
-> Result<(), AssertionError> {
    let triangle = right_triangle();
    let key = |i| Key {
        first: Feature::Face(0),
        second: Feature::Face(i),
    };
    for position in [
        point(10.0, 0.0, 0.0),
        point(0.0, 10.0, 0.0),
        point(32.0, 32.0, 0.0),
        point(10.0, -1e-14, 0.0),
        point(-1e-14, 10.0, 0.0),
        point(32.0, 32.0 + 1e-13, 0.0),
    ] {
        let found = run(&triangle, &[(key(0), position.clone())], &[], &[]).unwrap();
        let [u, v] = found.input.points[3];
        assert!(u > 0.0 && v > 0.0 && u + v < 1.0, "{position:?} {u} {v}");
        let [x, y] = [position[0].value() / 64.0, position[1].value() / 64.0];
        let tols = Assert {
            abs_tol: 1e-9,
            rel_tol: 1e-9,
            ..Assert::default()
        };
        tols.eq_within_tols(u, &x)?;
        tols.eq_within_tols(v, &y)?;
    }
    Ok(())
}

#[test]
fn degenerate_triangles_fail() {
    let line = [
        point(0.0, 0.0, 0.0),
        point(1.0, 1.0, 1.0),
        point(2.0, 2.0, 2.0),
    ];
    assert!(run(&line, &[], &[], &[]).is_err());
}

#[test]
fn segments_are_joined_through_the_corners_and_the_points_given() {
    let triangle = right_triangle();
    let planes = vec![plane(0, 20.0)];
    let inside = Key {
        first: Feature::Face(0),
        second: Feature::Face(0),
    };
    let corner = Key {
        first: Feature::Vertex(11),
        second: Feature::Face(0),
    };
    let side = edge_key(2, 0);
    let side_position = point(0.0, 20.0, 0.0);
    let points = vec![
        (inside.clone(), point(10.0, 10.0, 0.0)),
        (side.clone(), side_position),
        (corner.clone(), point(64.0, 0.0, 0.0)),
    ];
    let found = run(
        &triangle,
        &points,
        &[
            [corner.clone(), inside.clone()],
            [inside.clone(), side.clone()],
        ],
        &planes,
    )
    .unwrap();
    assert_eq!(found.keys.len(), 2);
    let at = |key: &Key| found.keys.iter().position(|k| k == key).unwrap() + 3;
    assert_eq!(
        found.input.segments,
        vec![[1, at(&inside)], [at(&inside), at(&side)]]
    );
    assert_eq!(found.input.sides[2], vec![at(&side)]);
    let triangles = triangulate(&found.input).unwrap();
    assert_eq!(triangles.len(), 2 * 5 - 4 - 2);
}

#[test]
fn a_segment_to_a_point_that_was_not_given_fails() {
    let triangle = right_triangle();
    let known = Key {
        first: Feature::Face(0),
        second: Feature::Face(0),
    };
    let unknown = Key {
        first: Feature::Face(0),
        second: Feature::Face(1),
    };
    let points = vec![(known.clone(), point(10.0, 10.0, 0.0))];
    assert!(run(&triangle, &points, &[[known, unknown]], &[]).is_err());
}

#[test]
fn points_on_features_that_are_not_the_triangles_fail() {
    let triangle = right_triangle();
    let not_a_corner = Key {
        first: Feature::Vertex(99),
        second: Feature::Face(0),
    };
    let not_an_edge = Key {
        first: Feature::Edge([10, 99]),
        second: Feature::Face(0),
    };
    for key in [not_a_corner, not_an_edge] {
        assert!(run(&triangle, &[(key, garbage())], &[], &[plane(0, 1.0)]).is_err());
    }
}

#[test]
fn points_that_the_cutters_do_not_tell_apart_or_cannot_place_fail() {
    let triangle = right_triangle();
    let same = vec![plane(0, 5.0), plane(0, 5.0)];
    let points: Vec<_> = (0..2).map(|i| (edge_key(0, i), garbage())).collect();
    assert!(run(&triangle, &points, &[], &same).is_err());
    let parallel = vec![plane(0, 5.0), plane(1, 5.0)];
    assert!(run(&triangle, &points, &[], &parallel).is_err());
}

#[test]
fn triangles_sharing_an_edge_subdivide_it_the_same_way() {
    let first = right_triangle();
    let second = [
        point(64.0, 0.0, 0.0),
        point(0.0, 0.0, 0.0),
        point(0.0, -64.0, 0.0),
    ];
    let planes: Vec<Plane> = OFFSETS.iter().map(|&c| plane(0, c)).collect();
    let points: Vec<_> = (0..planes.len())
        .map(|i| (edge_key(0, i), garbage()))
        .collect();
    let a = run(&first, &points, &[], &planes).unwrap();
    let b = frame(
        [&second[0], &second[1], &second[2]],
        [11, 10, 13],
        Side::Second,
        &points
            .iter()
            .map(|(key, position)| {
                let swapped = Key {
                    first: key.second.clone(),
                    second: key.first.clone(),
                };
                (swapped, position.clone())
            })
            .collect::<Vec<_>>(),
        &[],
        cutter(&planes),
    )
    .unwrap();
    let order = |found: &Frame, side: Side| -> Vec<usize> {
        found.input.sides[0]
            .iter()
            .map(|&index| {
                let key = &found.keys[index - 3];
                match if side == Side::First {
                    &key.second
                } else {
                    &key.first
                } {
                    Feature::Face(i) => *i,
                    _ => unreachable!(),
                }
            })
            .collect()
    };
    let mut reversed = order(&b, Side::Second);
    reversed.reverse();
    assert_eq!(order(&a, Side::First), reversed);
}

#[test]
fn the_ranks_of_the_points_are_the_order_of_their_keys() {
    let triangle = right_triangle();
    let key = |i| Key {
        first: Feature::Face(0),
        second: Feature::Face(i),
    };
    let positions = [(3, 10.0), (1, 20.0), (2, 5.0)];
    let points: Vec<_> = positions
        .iter()
        .map(|&(i, x)| (key(i), point(x, 10.0, 0.0)))
        .collect();
    let found = run(&triangle, &points, &[], &[]).unwrap();
    assert_eq!(found.keys, vec![key(1), key(2), key(3)]);
    assert_eq!(found.input.points[3][0], 20.0 / 64.0);
    assert_eq!(found.input.points[4][0], 5.0 / 64.0);
}
