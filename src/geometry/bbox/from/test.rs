use crate::{
    geometry::{Coordinate, CoordinateList, Coordinates, CoordinatesRef, bbox::BoundingBox},
    math::Tensor,
};

const MIN: Coordinate<3, 1> = Coordinate::const_from([-1.0, -1.0, -1.0]);
const MAX: Coordinate<3, 1> = Coordinate::const_from([1.0, 1.0, 1.0]);

#[test]
fn from_coordinates_list() {
    let coordinates = CoordinateList::<_, 1, _>::from([[0.0, 0.0, 0.0], MIN.into(), MAX.into()]);
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
    let coordinates_0 = Coordinates::<_, 1>::from([[0.0, 0.0, 0.0], MIN.into(), MAX.into()]);
    let coordinates: CoordinatesRef<_, _> = coordinates_0.iter().collect();
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
    let coordinates = Coordinates::<_, 1>::from([[0.0, 0.0, 0.0], MIN.into(), MAX.into()]);
    let bounding_box = BoundingBox::from(coordinates);
    assert_eq!(
        BoundingBox {
            minimum: MIN,
            maximum: MAX,
        },
        bounding_box
    )
}
