use crate::geometry::{
    Coordinate,
    bbox::{BoundingBox, Unite},
};

pub const D: usize = 3;
pub const I: usize = 1;

pub const BBOX_1: BoundingBox<D, I> = BoundingBox {
    minimum: Coordinate::const_from([0.0, 0.0, 0.0]),
    maximum: Coordinate::const_from([1.0, 1.0, 1.0]),
};

pub const BBOX_2: BoundingBox<D, I> = BoundingBox {
    minimum: Coordinate::const_from([0.5, 0.5, 0.5]),
    maximum: Coordinate::const_from([2.0, 2.0, 2.0]),
};

pub const BBOX_1U2: BoundingBox<D, I> = BoundingBox {
    minimum: Coordinate::const_from([0.0, 0.0, 0.0]),
    maximum: Coordinate::const_from([2.0, 2.0, 2.0]),
};

#[test]
fn unite() {
    assert_eq!(BBOX_1.unite(BBOX_2), BBOX_1U2);
}

#[test]
fn unite_ref() {
    assert_eq!(BBOX_1.unite(&BBOX_2), BBOX_1U2);
}

#[test]
fn ref_unite() {
    assert_eq!((&BBOX_1).unite(BBOX_2), BBOX_1U2);
}

#[test]
fn ref_unite_ref() {
    assert_eq!((&BBOX_1).unite(&BBOX_2), BBOX_1U2);
}
