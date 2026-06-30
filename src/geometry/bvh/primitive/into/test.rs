use crate::geometry::{
    Coordinate,
    bbox::{
        BoundingBox,
        test::{BBOX_1, BBOX_1U2, BBOX_2, D},
    },
    bvh::primitive::Primitive,
};

const PRIMITIVE_1: Primitive<D> = Primitive {
    bounding_box: BBOX_1,
    centroid: Coordinate::const_from([0.5, 0.5, 0.5]),
    index: 1,
};

const PRIMITIVE_2: Primitive<D> = Primitive {
    bounding_box: BBOX_2,
    centroid: Coordinate::const_from([1.5, 1.5, 1.5]),
    index: 2,
};

#[test]
fn primitives_slice_into_bounding_box() {
    let primitives = vec![PRIMITIVE_1, PRIMITIVE_2];
    let bbox = BoundingBox::from(primitives.as_slice());
    assert_eq!(bbox, BBOX_1U2)
}
