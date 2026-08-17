use crate::geometry::{Coordinate, Direction, bvh::ray::Ray};

#[test]
fn direction_is_normalized() {
    let ray = Ray::from((
        Coordinate::const_from([0.0, 0.0, 0.0]),
        Coordinate::const_from([0.0, 0.0, 5.0]),
    ));
    assert_eq!(ray.direction(), &Direction::const_from([0.0, 0.0, 1.0]));
}
