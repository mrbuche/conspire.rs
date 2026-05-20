use crate::geometry::{Coordinate, bbox::{BoundingBox, Union}};

const BBOX_1: BoundingBox<3, 1> = BoundingBox {
    minimum: Coordinate::const_from([0.0, 0.0, 0.0]),
    maximum: Coordinate::const_from([1.0, 1.0, 1.0]),
};

const BBOX_2: BoundingBox<3, 1> = BoundingBox {
    minimum: Coordinate::const_from([0.5, 0.5, 0.5]),
    maximum: Coordinate::const_from([2.0, 2.0, 2.0]),
};

const BBOX_1U2: BoundingBox<3, 1> = BoundingBox {
    minimum: Coordinate::const_from([0.0, 0.0, 0.0]),
    maximum: Coordinate::const_from([2.0, 2.0, 2.0]),
};

#[test]
fn union() {
    assert_eq!(BBOX_1.union(BBOX_2), BBOX_1U2);
}

#[test]
fn union_ref() {
    assert_eq!(BBOX_1.union(&BBOX_2), BBOX_1U2);
}

#[test]
fn ref_union() {
    assert_eq!((&BBOX_1).union(BBOX_2), BBOX_1U2);
}

#[test]
fn ref_union_ref() {
    assert_eq!((&BBOX_1).union(&BBOX_2), BBOX_1U2);
}
