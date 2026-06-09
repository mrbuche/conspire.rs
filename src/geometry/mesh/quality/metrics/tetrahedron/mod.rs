use crate::{geometry::Coordinates, math::Scalar};
use std::f64::consts::SQRT_2;

const CORNERS: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

pub(super) fn jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_jacobian(&CORNERS, element, coordinates)
}

pub(super) fn scaled_jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_scaled_jacobian(&CORNERS, element, coordinates, SQRT_2)
}
