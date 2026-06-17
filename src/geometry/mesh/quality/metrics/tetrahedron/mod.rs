#[cfg(test)]
mod test;

use crate::{geometry::Coordinates, math::Scalar};
use std::f64::consts::SQRT_2;

const CORNERS: [[usize; 3]; 4] = [[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]];

const EDGES: [[usize; 2]; 6] = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]];

const FACES: [[usize; 3]; 4] = [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

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
    super::min_scaled_jacobian(&CORNERS, element, coordinates, SQRT_2)
}

pub(super) fn maximum_skew<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    FACES
        .iter()
        .map(|face| {
            super::triangle_skew(
                &coordinates[element[face[0]]],
                &coordinates[element[face[1]]],
                &coordinates[element[face[2]]],
            )
        })
        .fold(Scalar::NEG_INFINITY, Scalar::max)
}

pub(super) fn volume<const D: usize>(element: &[usize], coordinates: &Coordinates<D>) -> Scalar {
    super::tet_volume(
        &[element[0], element[1], element[2], element[3]],
        coordinates,
    )
    .abs()
}
