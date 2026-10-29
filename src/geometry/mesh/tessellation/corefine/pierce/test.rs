use super::{Local, Meeting, crossing};
use crate::{
    geometry::{
        Coordinate, bvh::closest_point_on_triangle, mesh::tessellation::corefine::Triangle,
    },
    math::{Tensor, assert::AssertionError},
    units::Length,
};

type Points = [Coordinate<3>; 3];

fn refs(points: &Points) -> Triangle<'_> {
    [&points[0], &points[1], &points[2]]
}

fn ends(first: &Points, second: &Points) -> Option<Vec<(Local, Local, [f64; 3])>> {
    match crossing(refs(first), refs(second)).unwrap() {
        Meeting::Segment(ends) => Some(
            ends.map(|end| {
                let [x, y, z] = [0, 1, 2].map(|axis| end.point[axis].value());
                (end.first, end.second, [x, y, z])
            })
            .to_vec(),
        ),
        _ => None,
    }
}

fn flat() -> Points {
    [
        [0.0, 0.0, 0.0].into(),
        [4.0, 0.0, 0.0].into(),
        [0.0, 4.0, 0.0].into(),
    ]
}

fn assert_ends(
    first: &Points,
    second: &Points,
    expected: &[(Local, Local)],
) -> Result<(), AssertionError> {
    let found = ends(first, second).unwrap();
    assert_eq!(found.len(), expected.len());
    expected.iter().for_each(|pair| {
        assert!(
            found.iter().any(|&(a, b, _)| (a, b) == *pair),
            "{pair:?} not in {found:?}"
        )
    });
    let swapped = ends(second, first).unwrap();
    assert_eq!(swapped.len(), expected.len());
    expected.iter().for_each(|&(a, b)| {
        assert!(
            swapped.iter().any(|&(c, d, _)| (c, d) == (b, a)),
            "swapped {:?} not in {swapped:?}",
            (b, a)
        )
    });
    found.iter().for_each(|&(_, _, point)| {
        let on = |triangle: &Points| {
            closest_point_on_triangle(&point.into(), &triangle[0], &triangle[1], &triangle[2])
        };
        [first, second].iter().for_each(|triangle| {
            assert!((&on(triangle) - &Coordinate::from(point)).norm() < Length::meters(1e-12))
        })
    });
    Ok(())
}

#[test]
fn an_edge_through_the_inside_of_the_other_triangle() -> Result<(), AssertionError> {
    let upright: Points = [
        [1.0, -1.0, -1.0].into(),
        [1.0, 5.0, -1.0].into(),
        [1.0, 2.0, 3.0].into(),
    ];
    assert_ends(
        &flat(),
        &upright,
        &[(Local::Edge(0), Local::Face), (Local::Edge(1), Local::Face)],
    )
}

#[test]
fn an_edge_meeting_an_edge_is_one_end_named_from_both_sides() -> Result<(), AssertionError> {
    let upright: Points = [
        [1.0, 0.0, -1.0].into(),
        [1.0, 0.0, 1.0].into(),
        [1.0, 2.0, 0.5].into(),
    ];
    assert_ends(
        &flat(),
        &upright,
        &[
            (Local::Edge(0), Local::Edge(0)),
            (Local::Face, Local::Edge(2)),
        ],
    )
}

#[test]
fn a_vertex_in_the_inside_of_the_other_triangle() -> Result<(), AssertionError> {
    let upright: Points = [
        [1.0, 1.0, 0.0].into(),
        [1.0, 2.0, 1.0].into(),
        [1.0, 3.0, -1.0].into(),
    ];
    assert_ends(
        &flat(),
        &upright,
        &[
            (Local::Face, Local::Vertex(0)),
            (Local::Face, Local::Edge(1)),
        ],
    )
}

#[test]
fn an_edge_lying_in_the_plane_of_the_other_triangle() -> Result<(), AssertionError> {
    let upright: Points = [
        [1.0, 0.5, 0.0].into(),
        [1.0, 2.0, 0.0].into(),
        [1.0, 1.0, 1.0].into(),
    ];
    assert_ends(
        &flat(),
        &upright,
        &[
            (Local::Face, Local::Vertex(0)),
            (Local::Face, Local::Vertex(1)),
        ],
    )
}

#[test]
fn a_vertex_meeting_a_vertex_and_an_edge_meeting_an_edge() -> Result<(), AssertionError> {
    let first: Points = [
        [1.0, 0.0, 0.0].into(),
        [4.0, 0.0, 0.0].into(),
        [1.0, 3.0, 0.0].into(),
    ];
    let second: Points = [
        [1.0, 0.0, 0.0].into(),
        [1.0, 0.0, 2.0].into(),
        [1.0, 2.0, -1.0].into(),
    ];
    assert_ends(
        &first,
        &second,
        &[
            (Local::Vertex(0), Local::Vertex(0)),
            (Local::Edge(2), Local::Edge(1)),
        ],
    )
}

#[test]
fn triangles_touching_at_one_point_or_apart_have_no_ends() {
    let flat = flat();
    let touching: Points = [
        [1.0, 1.0, 0.0].into(),
        [1.0, 1.0, 2.0].into(),
        [1.0, 2.0, 2.0].into(),
    ];
    let apart: Points = [
        [9.0, 1.0, -1.0].into(),
        [9.0, 2.0, 1.0].into(),
        [9.0, 3.0, -1.0].into(),
    ];
    assert!(ends(&flat, &touching).is_none());
    assert!(ends(&flat, &apart).is_none());
}
