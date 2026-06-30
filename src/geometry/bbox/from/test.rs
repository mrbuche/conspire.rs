use crate::{
    geometry::{Coordinate, CoordinateList, Coordinates, CoordinatesRef, bbox::BoundingBox},
    math::Tensor,
};

const MIN: Coordinate<3> = Coordinate::const_from([-1.0, -1.0, -1.0]);
const MAX: Coordinate<3> = Coordinate::const_from([1.0, 1.0, 1.0]);

#[test]
fn from_coordinates_list() {
    let coordinates = CoordinateList::<_, _>::from([[0.0, 0.0, 0.0], MIN.into(), MAX.into()]);
    let bounding_box = BoundingBox::from(coordinates);
    assert_eq!(
        BoundingBox {
            minimum: MIN,
            maximum: MAX,
        },
        bounding_box
    )
}

#[test]
fn from_coordinates_ref() {
    let coordinates_0 = Coordinates::<_>::from([[0.0, 0.0, 0.0], MIN.into(), MAX.into()]);
    let coordinates: CoordinatesRef<_> = coordinates_0.iter().collect();
    let bounding_box = BoundingBox::from(coordinates);
    assert_eq!(
        BoundingBox {
            minimum: MIN,
            maximum: MAX,
        },
        bounding_box
    )
}

#[test]
fn from_coordinates_vec() {
    let coordinates = Coordinates::<_>::from([[0.0, 0.0, 0.0], MIN.into(), MAX.into()]);
    let bounding_box = BoundingBox::from(coordinates);
    assert_eq!(
        BoundingBox {
            minimum: MIN,
            maximum: MAX,
        },
        bounding_box
    )
}
