use super::{super::Feature, Crossing, Key, chain};
use crate::{
    geometry::{
        Coordinate,
        mesh::{
            Tessellation,
            test::{cuboid, solid},
        },
        solid::SolidOracle,
    },
    math::{Tensor, assert::AssertionError},
    units::Length,
};

fn key(node: usize) -> Key {
    Key {
        first: Feature::Vertex(node),
        second: Feature::Face(0),
    }
}

fn segment(a: usize, b: usize) -> Crossing {
    Crossing {
        first: 0,
        second: 0,
        segment: [a, b].map(|node| [node as f64, 0.0, 0.0].into()),
        keys: [key(a), key(b)],
    }
}

fn positions(points: &[Coordinate<3>]) -> Vec<f64> {
    points.iter().map(|point| point[0].value()).collect()
}

#[test]
fn shuffled_segments_form_one_open_chain() {
    let polylines = chain(&[segment(2, 3), segment(0, 1), segment(1, 2)]).unwrap();
    assert_eq!(polylines.len(), 1);
    let polyline = &polylines[0];
    assert!(!polyline.closed);
    assert_eq!(polyline.crossings.len(), 3);
    let mut positions = positions(&polyline.points);
    if positions[0] > positions[3] {
        positions.reverse();
    }
    assert_eq!(positions, vec![0.0, 1.0, 2.0, 3.0]);
}

#[test]
fn segments_around_a_cycle_form_a_closed_loop() {
    let polylines = chain(&[segment(0, 1), segment(2, 0), segment(1, 2)]).unwrap();
    assert_eq!(polylines.len(), 1);
    assert!(polylines[0].closed);
    assert_eq!(polylines[0].points.len(), 3);
    assert_eq!(polylines[0].crossings.len(), 3);
}

#[test]
fn separate_curves_stay_separate() {
    let polylines = chain(&[
        segment(0, 1),
        segment(10, 11),
        segment(1, 2),
        segment(20, 21),
        segment(21, 22),
        segment(22, 20),
    ])
    .unwrap();
    assert_eq!(polylines.len(), 3);
    assert_eq!(
        polylines.iter().filter(|polyline| polyline.closed).count(),
        1
    );
    let mut used: Vec<usize> = polylines
        .iter()
        .flat_map(|polyline| polyline.crossings.clone())
        .collect();
    used.sort_unstable();
    assert_eq!(used, vec![0, 1, 2, 3, 4, 5]);
}

#[test]
fn branching_and_degenerate_segments_are_errors() {
    assert!(chain(&[segment(0, 1), segment(0, 2), segment(0, 3)]).is_err());
    assert!(chain(&[segment(4, 4)]).is_err());
    assert!(chain(&[]).unwrap().is_empty());
}

fn assert_curves(first: &Tessellation, second: &Tessellation, loops: usize) {
    let crossings = first.crossings(second).unwrap();
    let polylines = first.intersection_curves(second).unwrap();
    assert_eq!(polylines.len(), loops);
    assert!(polylines.iter().all(|polyline| polyline.closed));
    let mut used: Vec<usize> = polylines
        .iter()
        .flat_map(|polyline| polyline.crossings.clone())
        .collect();
    used.sort_unstable();
    assert_eq!(used, (0..crossings.len()).collect::<Vec<_>>());
    polylines.iter().for_each(|polyline| {
        assert_eq!(polyline.points.len(), polyline.crossings.len());
        let length = polyline.points.len();
        (0..length).for_each(|index| {
            let (start, end) = (
                &polyline.points[index],
                &polyline.points[(index + 1) % length],
            );
            let ends = &crossings[polyline.crossings[index]].segment;
            let tolerance = Length::meters(1e-12);
            let joins = |a: &Coordinate<3>, b: &Coordinate<3>| (a - b).norm() < tolerance;
            assert!(
                (joins(start, &ends[0]) && joins(end, &ends[1]))
                    || (joins(start, &ends[1]) && joins(end, &ends[0]))
            );
        })
    });
}

fn assert_on_both(first: [[f64; 3]; 2], second: [[f64; 3]; 2]) -> Result<(), AssertionError> {
    let (a, b) = (solid(first[0], first[1]), solid(second[0], second[1]));
    a.tessellation()
        .intersection_curves(b.tessellation())?
        .iter()
        .flat_map(|polyline| polyline.points.iter())
        .for_each(|point| {
            assert!(a.signed_distance(point).abs() < Length::meters(1e-12));
            assert!(b.signed_distance(point).abs() < Length::meters(1e-12));
        });
    Ok(())
}

#[test]
fn boxes_overlapping_at_a_corner_meet_in_one_loop() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([0.37, 0.52, 0.29], [1.41, 1.33, 1.58]);
    assert_curves(&first, &second, 1);
}

#[test]
fn a_bar_through_a_box_meets_it_in_two_loops() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([0.3, 0.45, -0.5], [0.6, 0.72, 1.5]);
    assert_curves(&first, &second, 2);
}

#[test]
fn curve_points_lie_on_both_surfaces() -> Result<(), AssertionError> {
    assert_on_both(
        [[0.0; 3], [1.0; 3]],
        [[0.37, 0.52, 0.29], [1.41, 1.33, 1.58]],
    )?;
    assert_on_both([[0.0; 3], [1.0; 3]], [[0.3, 0.45, -0.5], [0.6, 0.72, 1.5]])
}

#[test]
fn crossings_on_a_shared_diagonal_still_close_into_the_skew_hexagon() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([0.5; 3], [1.5; 3]);
    let polylines = first.intersection_curves(&second).unwrap();
    assert_eq!(polylines.len(), 1);
    assert!(polylines[0].closed);
    assert_eq!(polylines[0].crossings.len(), 6);
    let mut found: Vec<[f64; 3]> = polylines[0]
        .points
        .iter()
        .map(|point| [point[0].value(), point[1].value(), point[2].value()])
        .collect();
    let mut expected = vec![
        [1.0, 0.5, 0.5],
        [1.0, 0.5, 1.0],
        [0.5, 0.5, 1.0],
        [0.5, 1.0, 1.0],
        [0.5, 1.0, 0.5],
        [1.0, 1.0, 0.5],
    ];
    found.sort_by(|a, b| a.partial_cmp(b).unwrap());
    expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(found.len(), 6);
    found.iter().zip(&expected).for_each(|(a, b)| {
        a.iter()
            .zip(b)
            .for_each(|(x, y)| assert!((x - y).abs() < 1e-12, "{found:?}"))
    });
}

#[test]
fn a_diagonal_shift_of_a_cube_closes_into_one_loop_at_every_offset() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    [0.25, 0.5, 0.75].iter().for_each(|&shift| {
        let second = cuboid([shift; 3], [shift + 1.0; 3]);
        let polylines = first.intersection_curves(&second).unwrap();
        assert_eq!(polylines.len(), 1, "shift {shift}");
        assert!(polylines[0].closed, "shift {shift}");
    })
}

#[test]
fn a_bar_whose_edges_lie_on_face_diagonals_still_closes() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([0.3, 0.3, 0.5], [0.7, 0.7, 1.5]);
    let polylines = first.intersection_curves(&second).unwrap();
    assert_eq!(polylines.len(), 1);
    assert!(polylines[0].closed);
    polylines[0]
        .points
        .iter()
        .for_each(|point| assert!((point[2].value() - 1.0).abs() < 1e-12));
    [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]]
        .iter()
        .for_each(|corner| {
            assert!(polylines[0].points.iter().any(|point| {
                (point[0].value() - corner[0]).abs() < 1e-12
                    && (point[1].value() - corner[1]).abs() < 1e-12
            }))
        });
}

#[test]
fn disjoint_tessellations_have_no_curves() {
    let first = cuboid([0.0; 3], [1.0; 3]);
    let second = cuboid([3.0; 3], [4.0; 3]);
    assert!(first.intersection_curves(&second).unwrap().is_empty());
}
