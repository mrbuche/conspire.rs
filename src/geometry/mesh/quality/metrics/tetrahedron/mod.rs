#[cfg(test)]
mod test;

use crate::{geometry::Coordinates, math::Scalar};
use std::f64::consts::SQRT_2;

const CORNERS: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

const EDGES: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

pub(super) fn minimum_edge_ratio(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::minimum_edge_ratio(&EDGES, element, coordinates)
}

pub(super) fn minimum_jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_jacobian(&CORNERS, element, coordinates)
}

pub(super) fn minimum_scaled_jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_scaled_jacobian(&CORNERS, element, coordinates, SQRT_2)
}