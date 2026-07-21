#[cfg(test)]
mod test;

use crate::{
    geometry::Coordinates,
    math::{Scalar, Tensor},
};

pub(crate) const CORNERS: [[usize; 3]; 8] = [
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
    let x1 = (p(1) - p(0)) + (p(2) - p(3)) + (p(5) - p(4)) + (p(6) - p(7));
    let x2 = (p(3) - p(0)) + (p(2) - p(1)) + (p(7) - p(4)) + (p(6) - p(5));
    let x3 = (p(4) - p(0)) + (p(5) - p(1)) + (p(6) - p(2)) + (p(7) - p(3));
    [(&x1, &x2), (&x1, &x3), (&x2, &x3)]
        .into_iter()
        .map(|(u, v)| {
            let (nu, nv) = (u.norm(), v.norm());
            if nu > 0.0 && nv > 0.0 {
                ((u * v) / (nu * nv)).abs()
            } else {
                0.0
            }
        })
        .fold(Scalar::NEG_INFINITY, Scalar::max)
}

pub(super) fn volume<const D: usize>(element: &[usize], coordinates: &Coordinates<D>) -> Scalar {
    let p = |i: usize| &coordinates[element[i]];
    let x1 = (p(1) - p(0)) + (p(2) - p(3)) + (p(5) - p(4)) + (p(6) - p(7));
    let x2 = (p(3) - p(0)) + (p(2) - p(1)) + (p(7) - p(4)) + (p(6) - p(5));
    let x3 = (p(4) - p(0)) + (p(5) - p(1)) + (p(6) - p(2)) + (p(7) - p(3));
    let x2_cross_x3 = super::cross(&x2, &x3);
    (x1[0] * x2_cross_x3[0] + x1[1] * x2_cross_x3[1] + x1[2] * x2_cross_x3[2]) / 64.0
}
