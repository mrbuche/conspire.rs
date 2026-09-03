#[cfg(test)]
mod test;

use crate::{
    geometry::Coordinates,
    math::{Quantity, Scalar},
    units::Area,
};

const CORNERS: [(usize, [usize; 2]); 4] = [(0, [1, 3]), (1, [2, 0]), (2, [3, 1]), (3, [0, 2])];

const EDGES: [[usize; 2]; 4] = [[0, 1], [1, 2], [2, 3], [3, 0]];

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
    super::min_scaled_jacobian(&CORNERS, element, coordinates, 1.0)
}

pub(super) fn maximum_skew<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    super::quad_skew(
        &coordinates[element[0]],
        &coordinates[element[1]],
        &coordinates[element[2]],
        &coordinates[element[3]],
    )
}

pub(super) fn volume<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Quantity<Area> {
    super::triangle_area(&[element[0], element[1], element[2]], coordinates)
        + super::triangle_area(&[element[0], element[2], element[3]], coordinates)
}
