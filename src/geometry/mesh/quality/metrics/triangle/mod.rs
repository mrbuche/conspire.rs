#[cfg(test)]
mod test;

use crate::{geometry::Coordinates, math::Scalar};

const CORNERS: [[usize; 2]; 3] = [[1, 2], [2, 0], [0, 1]];

const EDGES: [[usize; 2]; 3] = [[0, 1], [1, 2], [2, 0]];

pub(super) fn maximum_edge_ratio<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    super::maximum_edge_ratio(&EDGES, element, coordinates)
}

pub(super) fn minimum_jacobian<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    super::min_jacobian(&CORNERS, element, coordinates)
}

pub(super) fn minimum_scaled_jacobian<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    super::min_scaled_jacobian(&CORNERS, element, coordinates, 2.0 / 3.0_f64.sqrt())
}
