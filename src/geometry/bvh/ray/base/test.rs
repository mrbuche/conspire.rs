use crate::geometry::{Coordinate, bbox::test::BBOX_1, bvh::ray::Ray};

#[test]
fn hits_box_from_outside() {
    let ray = Ray::from((
        Coordinate::const_from([0.5, 0.5, -2.0]),
        Coordinate::const_from([0.0, 0.0, 1.0]),
    ));
    assert_eq!(ray.intersects(&BBOX_1), Some(2.0));
}

#[test]
fn origin_inside_box_returns_zero() {
    let ray = Ray::from((
        Coordinate::const_from([0.5, 0.5, 0.5]),
        Coordinate::const_from([0.0, 0.0, 1.0]),
    ));
    assert_eq!(ray.intersects(&BBOX_1), Some(0.0));
}

#[test]
fn pointing_away_misses() {
    let ray = Ray::from((
        Coordinate::const_from([0.5, 0.5, -2.0]),
        Coordinate::const_from([0.0, 0.0, -1.0]),
    ));
    assert_eq!(ray.intersects(&BBOX_1), None);
}

#[test]
fn parallel_outside_misses() {
    let ray = Ray::from((
        Coordinate::const_from([5.0, 5.0, -2.0]),
        Coordinate::const_from([0.0, 0.0, 1.0]),
    ));
    assert_eq!(ray.intersects(&BBOX_1), None);
}
