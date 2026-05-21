use crate::geometry::bbox::unite::test::{BBOX_1, BBOX_2, D};

#[test]
fn either_axis_cube() {
    assert_eq!(BBOX_1.longest_axis(), D - 1);
    assert_eq!(BBOX_1.shortest_axis(), 0)
}

#[test]
fn either_axis_other() {
    assert_eq!(BBOX_2.longest_axis(), 1);
    assert_eq!(BBOX_2.shortest_axis(), 2)
}
