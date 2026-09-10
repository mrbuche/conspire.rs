#[cfg(test)]
mod test;

use crate::{
    geometry::Coordinates,
    math::{Quantity, Scalar},
    units::Volume,
};
use std::f64::consts::SQRT_2;

pub(crate) const CORNERS: [(usize, [usize; 3]); 8] = [
    (0, [1, 3, 4]),
    (1, [2, 0, 4]),
    (2, [3, 1, 4]),
    (3, [0, 2, 4]),
    (4, [0, 2, 1]),
    (4, [1, 3, 2]),
    (4, [2, 0, 3]),
    (4, [3, 1, 0]),
];

const EDGES: [[usize; 2]; 8] = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 0],
    [0, 4],
    [1, 4],
    [2, 4],
    [3, 4],
];

const TRIANGLES: [[usize; 3]; 4] = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]];

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
    let base = super::quad_skew(
        &coordinates[element[0]],
        &coordinates[element[1]],
        &coordinates[element[2]],
        &coordinates[element[3]],
    );
    TRIANGLES
        .iter()
        .map(|face| {
            super::triangle_skew(
                &coordinates[element[face[0]]],
                &coordinates[element[face[1]]],
                &coordinates[element[face[2]]],
            )
        })
        .fold(base, Scalar::max)
}

pub(super) fn volume<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Quantity<Volume> {
    super::tet_volume(
        &[element[0], element[1], element[2], element[4]],
        coordinates,
    ) + super::tet_volume(
        &[element[0], element[2], element[3], element[4]],
        coordinates,
    )
}
