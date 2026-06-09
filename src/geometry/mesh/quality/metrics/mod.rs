mod hexahedron;
mod quadrilateral;
mod tetrahedron;
mod triangle;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::{Scalar, Tensor, TensorRank2},
};
use std::array::from_fn;

pub trait Verdict {
    fn maximum_edge_ratios(&self) -> Vec<Vec<Scalar>>;
    fn minimum_jacobians(&self) -> Vec<Vec<Scalar>>;
    fn minimum_scaled_jacobians(&self) -> Vec<Vec<Scalar>>;
}

impl<const D: usize> Verdict for Mesh<D> {
    fn maximum_edge_ratios(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| match block {
                Connectivity::Triangular(elements) => elements
                    .iter()
                    .map(|element| triangle::maximum_edge_ratio(element, coordinates))
                    .collect(),
                Connectivity::Quadrilateral(elements) => elements
                    .iter()
                    .map(|element| quadrilateral::maximum_edge_ratio(element, coordinates))
                    .collect(),
                Connectivity::Tetrahedral(elements) => elements
                    .iter()
                    .map(|element| tetrahedron::maximum_edge_ratio(element, coordinates))
                    .collect(),
                Connectivity::Hexahedral(elements) => elements
                    .iter()
                    .map(|element| hexahedron::maximum_edge_ratio(element, coordinates))
                    .collect(),
                _ => todo!(),
            })
            .collect()
    }
    fn minimum_jacobians(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| match block {
                Connectivity::Triangular(elements) => elements
                    .iter()
                    .map(|element| triangle::minimum_jacobian(element, coordinates))
                    .collect(),
                Connectivity::Quadrilateral(elements) => elements
                    .iter()
                    .map(|element| quadrilateral::minimum_jacobian(element, coordinates))
                    .collect(),
                Connectivity::Tetrahedral(elements) => elements
                    .iter()
                    .map(|element| tetrahedron::minimum_jacobian(element, coordinates))
                    .collect(),
                Connectivity::Hexahedral(elements) => elements
                    .iter()
                    .map(|element| hexahedron::minimum_jacobian(element, coordinates))
                    .collect(),
                _ => todo!(),
            })
            .collect()
    }
    fn minimum_scaled_jacobians(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| match block {
                Connectivity::Triangular(elements) => elements
                    .iter()
                    .map(|element| triangle::minimum_scaled_jacobian(element, coordinates))
                    .collect(),
                Connectivity::Quadrilateral(elements) => elements
                    .iter()
                    .map(|element| quadrilateral::minimum_scaled_jacobian(element, coordinates))
                    .collect(),
                Connectivity::Tetrahedral(elements) => elements
                    .iter()
                    .map(|element| tetrahedron::minimum_scaled_jacobian(element, coordinates))
                    .collect(),
                Connectivity::Hexahedral(elements) => elements
                    .iter()
                    .map(|element| hexahedron::minimum_scaled_jacobian(element, coordinates))
                    .collect(),
                _ => todo!(),
            })
            .collect()
    }
}

fn maximum_edge_ratio<const D: usize, const E: usize>(
    edges: &[[usize; 2]; E],
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    let mut shortest = Scalar::INFINITY;
    let mut longest: Scalar = 0.0;
    for [a, b] in edges {
        let length = (&coordinates[element[*b]] - &coordinates[element[*a]]).norm();
        shortest = shortest.min(length);
        longest = longest.max(length);
    }
    if shortest > 0.0 {
        longest / shortest
    } else {
        Scalar::INFINITY
    }
}

fn min_jacobian<const D: usize, const K: usize, const C: usize>(
    table: &[[usize; K]; C],
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    corners(table, element, coordinates)
        .into_iter()
        .map(|(measure, _)| measure)
        .fold(Scalar::INFINITY, Scalar::min)
}

fn min_scaled_jacobian<const D: usize, const K: usize, const C: usize>(
    table: &[[usize; K]; C],
    element: &[usize],
    coordinates: &Coordinates<D>,
    scale: Scalar,
) -> Scalar {
    corners(table, element, coordinates)
        .into_iter()
        .map(|(measure, normalizer)| {
            if normalizer > 0.0 {
                (scale * measure / normalizer).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .fold(Scalar::INFINITY, Scalar::min)
}

fn corners<const D: usize, const K: usize, const C: usize>(
    table: &[[usize; K]; C],
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> [(Scalar, Scalar); C] {
    from_fn(|corner| {
        let origin = &coordinates[element[corner]];
        let edges: [Coordinate<D>; K] =
            from_fn(|edge| &coordinates[element[table[corner][edge]]] - origin);
        let normalizer: Scalar = edges.iter().map(|edge| edge.norm()).product();
        (corner_measure(&edges), normalizer)
    })
}

fn corner_measure<const D: usize, const K: usize>(edges: &[Coordinate<D>; K]) -> Scalar {
    if K == D {
        let matrix: [[Scalar; K]; K] = from_fn(|row| from_fn(|column| edges[row][column]));
        TensorRank2::<K, 0, 0>::from(matrix).determinant()
    } else {
        let gram: [[Scalar; K]; K] = from_fn(|i| from_fn(|j| &edges[i] * &edges[j]));
        TensorRank2::<K, 0, 0>::from(gram)
            .determinant()
            .max(0.0)
            .sqrt()
    }
}

#[derive(Clone, Copy)]
pub(super) enum Kind {
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
}

impl Kind {
    pub(super) fn of(connectivity: &Connectivity) -> Option<Self> {
        match connectivity {
            Connectivity::Triangular(_) => Some(Self::Triangle),
            Connectivity::Quadrilateral(_) => Some(Self::Quadrilateral),
            Connectivity::Tetrahedral(_) => Some(Self::Tetrahedron),
            Connectivity::Hexahedral(_) => Some(Self::Hexahedron),
            _ => None,
        }
    }
}

pub(super) fn minimum_jacobian<const D: usize>(
    kind: Kind,
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    match kind {
        Kind::Triangle => triangle::minimum_jacobian(element, coordinates),
        Kind::Quadrilateral => quadrilateral::minimum_jacobian(element, coordinates),
        Kind::Tetrahedron => tetrahedron::minimum_jacobian(element, coordinates),
        Kind::Hexahedron => hexahedron::minimum_jacobian(element, coordinates),
    }
}

pub(super) fn minimum_scaled_jacobian<const D: usize>(
    kind: Kind,
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    match kind {
        Kind::Triangle => triangle::minimum_scaled_jacobian(element, coordinates),
        Kind::Quadrilateral => quadrilateral::minimum_scaled_jacobian(element, coordinates),
        Kind::Tetrahedron => tetrahedron::minimum_scaled_jacobian(element, coordinates),
        Kind::Hexahedron => hexahedron::minimum_scaled_jacobian(element, coordinates),
    }
}
