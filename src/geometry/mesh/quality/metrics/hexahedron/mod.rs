#[cfg(test)]
mod test;

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

const EDGES: [[usize; 2]; 12] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [4, 5],
    [5, 6],
    [6, 7],
    [7, 4],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
];

pub(super) fn minimum_edge_ratio(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::minimum_edge_ratio(&EDGES, element, coordinates)
}

pub(super) fn minimum_jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_jacobian(&CORNERS, element, coordinates)
}

pub(super) fn minimum_scaled_jacobian(element: &[usize], coordinates: &Coordinates<3>) -> Scalar {
    super::min_scaled_jacobian(&CORNERS, element, coordinates, 1.0)
}