use crate::geometry::{Coordinate, bbox::BoundingBox};

pub const D: usize = 3;

pub const BBOX_1: BoundingBox<D> = BoundingBox {
    minimum: Coordinate::const_from([0.0, 0.0, 0.0]),
    maximum: Coordinate::const_from([1.0, 1.0, 1.0]),
};

pub const BBOX_2: BoundingBox<D> = BoundingBox {
    minimum: Coordinate::const_from([0.25, 0.0, 0.5]),
    maximum: Coordinate::const_from([2.0, 2.0, 2.0]),
};

pub const BBOX_1U2: BoundingBox<D> = BoundingBox {
    minimum: Coordinate::const_from([0.0, 0.0, 0.0]),
    maximum: Coordinate::const_from([2.0, 2.0, 2.0]),
};
