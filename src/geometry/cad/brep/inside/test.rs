use super::point_in_polygon;
use crate::geometry::{
    Coordinate,
    cad::brep::test::{axis_aligned_box, unit_cube},
};

#[test]
fn square_with_a_hole() {
    let rings = vec![
        vec![[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
        vec![[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]],
    ];
    assert!(point_in_polygon([0.5, 0.5], &rings));
    assert!(point_in_polygon([3.0, 3.0], &rings));
    assert!(!point_in_polygon([1.5, 1.5], &rings)); // in the hole
    assert!(!point_in_polygon([5.0, 2.0], &rings)); // outside
}

#[test]
fn box_interior_and_exterior() {
    let brep = axis_aligned_box([2.0, 4.0, 8.0]);
    let inside = [[1.0, 2.0, 4.0], [0.01, 0.01, 0.01], [1.99, 3.99, 7.99]];
    let outside = [
        [3.0, 2.0, 4.0],
        [-1.0, 2.0, 4.0],
        [1.0, 2.0, 10.0],
        [1.0, -0.5, 4.0],
    ];
    for point in inside {
        assert!(
            brep.encloses(&Coordinate::from(point)).unwrap(),
            "{point:?}"
        );
    }
    for point in outside {
        assert!(
            !brep.encloses(&Coordinate::from(point)).unwrap(),
            "{point:?}"
        );
    }
}

#[test]
fn unit_cube_interior_and_exterior() {
    let brep = unit_cube();
    assert!(brep.encloses(&Coordinate::from([0.5, 0.5, 0.5])).unwrap());
    assert!(!brep.encloses(&Coordinate::from([0.5, 0.5, 1.5])).unwrap());
}
