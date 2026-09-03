use super::mixed_point_in_polygon;
use crate::geometry::{
    Coordinate,
    cad::brep::{
        planar::Arc2,
        test::{axis_aligned_box, unit_cube},
    },
};

#[test]
fn square_with_a_hole() {
    let straight = |points: &[[f64; 2]]| points.iter().map(|&point| (point, None)).collect();
    let rings = vec![
        straight(&[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]),
        straight(&[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]),
    ];
    assert!(mixed_point_in_polygon([0.5, 0.5], &rings));
    assert!(mixed_point_in_polygon([3.0, 3.0], &rings));
    assert!(!mixed_point_in_polygon([1.5, 1.5], &rings)); // in the hole
    assert!(!mixed_point_in_polygon([5.0, 2.0], &rings)); // outside
}

#[test]
fn a_tangent_ray_does_not_flip_parity() {
    // Upper half-disk: diameter b->a on y = 0, semicircular arc a->b (ccw) on
    // top. Ring entry i carries the arc for edge i -> i+1.
    let ring = vec![vec![
        (
            [2.0, 0.0],
            Some(Arc2 {
                centre: [0.0, 0.0],
                radius: 2.0,
                ccw: true,
            }),
        ),
        ([-2.0, 0.0], None),
    ]];
    // Far left at y = 2 exactly: the +x ray is tangent to the arc's apex. The
    // point is outside; a tangent must contribute zero crossings, not one.
    assert!(!mixed_point_in_polygon([-5.0, 2.0], &ring));
    // Genuinely inside still reads inside.
    assert!(mixed_point_in_polygon([0.0, 1.0], &ring));
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
