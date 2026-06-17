#[cfg(test)]
mod test;

use crate::{
    geometry::Coordinates,
    math::{Scalar, Tensor},
};

const CORNERS: [[usize; 2]; 4] = [[1, 3], [2, 0], [3, 1], [0, 2]];

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
    let p = |i: usize| &coordinates[element[i]];
    let x1 = (p(1) - p(0)) + (p(2) - p(3));
    let x2 = (p(3) - p(0)) + (p(2) - p(1));
    let (n1, n2) = (x1.norm(), x2.norm());
    if n1 > 0.0 && n2 > 0.0 {
        ((&x1 * &x2) / (n1 * n2)).abs()
    } else {
        0.0
    }
}

pub(super) fn volume<const D: usize>(element: &[usize], coordinates: &Coordinates<D>) -> Scalar {
    super::triangle_area(&[element[0], element[1], element[2]], coordinates)
        + super::triangle_area(&[element[0], element[2], element[3]], coordinates)
}
