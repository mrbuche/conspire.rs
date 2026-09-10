#[cfg(test)]
mod test;

use crate::{
    geometry::Coordinates,
    math::{Quantity, Scalar},
    units::Volume,
};

pub(crate) const CORNERS: [(usize, [usize; 3]); 6] = [
    (0, [1, 2, 3]),
    (1, [2, 0, 4]),
    (2, [0, 1, 5]),
    (3, [5, 4, 0]),
    (4, [3, 5, 1]),
    (5, [4, 3, 2]),
];

const EDGES: [[usize; 2]; 9] = [
    [0, 1],
    [1, 2],
    [2, 0],
    [3, 4],
    [4, 5],
    [5, 3],
    [0, 3],
    [1, 4],
    [2, 5],
];

const TRIANGLES: [[usize; 3]; 2] = [[0, 1, 2], [3, 4, 5]];

const QUADRILATERALS: [[usize; 4]; 3] = [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]];

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

pub(super) fn maximum_skew<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    let triangles = TRIANGLES.iter().map(|face| {
        super::triangle_skew(
            &coordinates[element[face[0]]],
            &coordinates[element[face[1]]],
            &coordinates[element[face[2]]],
        )
    });
    let quadrilaterals = QUADRILATERALS.iter().map(|face| {
        super::quad_skew(
            &coordinates[element[face[0]]],
            &coordinates[element[face[1]]],
            &coordinates[element[face[2]]],
            &coordinates[element[face[3]]],
        )
    });
    triangles
        .chain(quadrilaterals)
        .fold(Scalar::NEG_INFINITY, Scalar::max)
}

pub(super) fn volume<const D: usize>(
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Quantity<Volume> {
    super::tet_volume(
        &[element[0], element[1], element[2], element[5]],
        coordinates,
    ) + super::tet_volume(
        &[element[0], element[1], element[5], element[4]],
        coordinates,
    ) + super::tet_volume(
        &[element[0], element[4], element[5], element[3]],
        coordinates,
    )
}
