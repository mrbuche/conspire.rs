use crate::geometry::{Coordinate, CoordinateList, Coordinates, bbox::BoundingBox};

const MIN: Coordinate<3, 1> = Coordinate::const_from([-1.0, -1.0, -1.0]);
const MAX: Coordinate<3, 1> = Coordinate::const_from([1.0, 1.0, 1.0]);

#[test]
fn from_coordinate_list() {
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
fn from_coordinate_vec() {
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
