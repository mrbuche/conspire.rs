use crate::geometry::{Coordinate, Coordinates, bbox::BoundingBox, bvh::primitive::Primitive};

const CENTROID: Coordinate<3, 0> = Coordinate::const_from([1.0, 2.0, 3.0]);
const INDEX: usize = 3;

fn bbox() -> BoundingBox<3, 0> {
    BoundingBox::from(Coordinates::from([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]]))
}

fn primitive() -> Primitive<3, 0, usize> {
    Primitive {
        bounding_box: bbox(),
        centroid: CENTROID,
        index: INDEX,
    }
}

#[test]
fn bounding_box() {
    assert_eq!(primitive().bounding_box(), &bbox())
}

#[test]
fn centroid() {
    assert_eq!(primitive().centroid(), &CENTROID)
}

#[test]
fn index() {
    assert_eq!(primitive().index(), INDEX)
}
