use crate::{geometry::Coordinates, math::Scalar};

const CORNERS: [[usize; 3]; 8] = [
    [1, 3, 4],
    [2, 0, 5],
    [3, 1, 6],
    [0, 2, 7],
    [7, 5, 0],
    [4, 6, 1],
    [5, 7, 2],
    [6, 4, 3],
];

pub(super) fn jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_jacobian(&CORNERS, element, coordinates)
}

pub(super) fn scaled_jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_scaled_jacobian(&CORNERS, element, coordinates, 1.0)
}
