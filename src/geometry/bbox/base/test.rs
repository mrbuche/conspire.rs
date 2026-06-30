use crate::geometry::bbox::test::{BBOX_1, BBOX_2, D};

#[test]
fn either_axis_cube() {
    assert_eq!(BBOX_1.minimum(), &[0.0; D].into());
    assert_eq!(BBOX_1.maximum(), &[1.0; D].into());
    assert_eq!(BBOX_1.longest_axis(), D - 1);
    assert_eq!(BBOX_1.shortest_axis(), 0)
}

#[test]
fn either_axis_other() {
    assert_eq!(BBOX_2.longest_axis(), 1);
    assert_eq!(BBOX_2.shortest_axis(), 2)
}

#[test]
fn triangle_slices_through_without_a_vertex_inside() {
    let (a, b, c) = (
        [-1.0, -1.0, 0.5].into(),
        [3.0, -1.0, 0.5].into(),
        [-1.0, 3.0, 0.5].into(),
    );
    assert!(BBOX_1.overlaps_triangle(&a, &b, &c));
}

#[test]
fn triangle_far_away_does_not_overlap() {
    let (a, b, c) = (
        [0.0, 0.0, 5.0].into(),
        [1.0, 0.0, 5.0].into(),
        [0.0, 1.0, 5.0].into(),
    );
    assert!(!BBOX_1.overlaps_triangle(&a, &b, &c));
}

#[test]
fn triangle_beside_box_in_plane_does_not_overlap() {
    let (a, b, c) = (
        [5.0, 5.0, 0.5].into(),
        [6.0, 5.0, 0.5].into(),
        [5.0, 6.0, 0.5].into(),
    );
    assert!(!BBOX_1.overlaps_triangle(&a, &b, &c));
}
