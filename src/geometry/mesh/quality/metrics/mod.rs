#[cfg(test)]
mod test;

pub(crate) mod hexahedron;
mod quadrilateral;
mod tetrahedron;
mod triangle;

use crate::{
    geometry::{
        Coordinate, Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::{Quantity, Reference, Scalar, Tensor, TensorRank1List, TensorRank2},
    units::{Area, Length, Volume},
};
use std::array::from_fn;

const EQUIANGLE: Scalar = std::f64::consts::FRAC_PI_3;

pub trait Verdict {
    fn maximum_edge_ratios(&self) -> Vec<Vec<Scalar>>;
    fn maximum_skews(&self) -> Vec<Vec<Scalar>>;
    fn minimum_jacobians(&self) -> Vec<Vec<Scalar>>;
    fn minimum_scaled_jacobians(&self) -> Vec<Vec<Scalar>>;
    fn volumes(&self) -> Vec<Vec<Scalar>>;
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
                Connectivity::Polygonal(_)
                | Connectivity::Polyhedral(_)
                | Connectivity::Pyramidal(_)
                | Connectivity::Wedge(_) => vec![Scalar::NAN; block.number_of_elements()],
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
                Connectivity::Polygonal(_)
                | Connectivity::Polyhedral(_)
                | Connectivity::Pyramidal(_)
                | Connectivity::Wedge(_) => vec![Scalar::NAN; block.number_of_elements()],
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
                Connectivity::Polygonal(_)
                | Connectivity::Polyhedral(_)
                | Connectivity::Pyramidal(_)
                | Connectivity::Wedge(_) => vec![Scalar::NAN; block.number_of_elements()],
            })
            .collect()
    }
    fn maximum_skews(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| match block {
                Connectivity::Triangular(elements) => elements
                    .iter()
                    .map(|element| triangle::maximum_skew(element, coordinates))
                    .collect(),
                Connectivity::Quadrilateral(elements) => elements
                    .iter()
                    .map(|element| quadrilateral::maximum_skew(element, coordinates))
                    .collect(),
                Connectivity::Tetrahedral(elements) => elements
                    .iter()
                    .map(|element| tetrahedron::maximum_skew(element, coordinates))
                    .collect(),
                Connectivity::Hexahedral(elements) => elements
                    .iter()
                    .map(|element| hexahedron::maximum_skew(element, coordinates))
                    .collect(),
                Connectivity::Polygonal(_)
                | Connectivity::Polyhedral(_)
                | Connectivity::Pyramidal(_)
                | Connectivity::Wedge(_) => vec![Scalar::NAN; block.number_of_elements()],
            })
            .collect()
    }
    fn volumes(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| match block {
                Connectivity::Triangular(elements) => elements
                    .iter()
                    .map(|element| triangle::volume(element, coordinates).value())
                    .collect(),
                Connectivity::Quadrilateral(elements) => elements
                    .iter()
                    .map(|element| quadrilateral::volume(element, coordinates).value())
                    .collect(),
                Connectivity::Tetrahedral(elements) => elements
                    .iter()
                    .map(|element| tetrahedron::volume(element, coordinates).value())
                    .collect(),
                Connectivity::Hexahedral(elements) => elements
                    .iter()
                    .map(|element| hexahedron::volume(element, coordinates).value())
                    .collect(),
                Connectivity::Polygonal(_)
                | Connectivity::Polyhedral(_)
                | Connectivity::Pyramidal(_)
                | Connectivity::Wedge(_) => vec![Scalar::NAN; block.number_of_elements()],
            })
            .collect()
    }
}

fn cross<const D: usize>(a: &Coordinate<D>, b: &Coordinate<D>) -> [Quantity<Area>; 3] {
    let zero = Quantity::<Length>::default();
    let az = if D > 2 { a[2] } else { zero };
    let bz = if D > 2 { b[2] } else { zero };
    [
        a[1] * bz - az * b[1],
        az * b[0] - a[0] * bz,
        a[0] * b[1] - a[1] * b[0],
    ]
}

pub(crate) fn chi(epsilon: Scalar, determinant: Scalar) -> Scalar {
    0.5 * (determinant + (epsilon * epsilon + determinant * determinant).sqrt())
}

pub(crate) fn regularized(edges: &TensorRank1List<3, Reference, 3>, epsilon: Scalar) -> Scalar {
    edges.norm_squared().value().powf(1.5) / chi(epsilon, edges.scalar_triple_product())
}

fn triple_product<const D: usize>(
    a: &Coordinate<D>,
    b: &Coordinate<D>,
    c: &Coordinate<D>,
) -> Quantity<Volume> {
    let bc = cross(b, c);
    a[0] * bc[0] + a[1] * bc[1] + a[2] * bc[2]
}

fn cross_norm<const D: usize>(a: &Coordinate<D>, b: &Coordinate<D>) -> Quantity<Area> {
    let n = cross(a, b);
    Quantity::new((n[0].value().powi(2) + n[1].value().powi(2) + n[2].value().powi(2)).sqrt())
}

fn triangle_area<const D: usize>(
    triangle: &[usize; 3],
    coordinates: &Coordinates<D>,
) -> Quantity<Area> {
    let a = &coordinates[triangle[1]] - &coordinates[triangle[0]];
    let b = &coordinates[triangle[2]] - &coordinates[triangle[0]];
    cross_norm(&a, &b) * 0.5
}

fn tet_volume<const D: usize>(
    tetrahedron: &[usize; 4],
    coordinates: &Coordinates<D>,
) -> Quantity<Volume> {
    let a = &coordinates[tetrahedron[1]] - &coordinates[tetrahedron[0]];
    let b = &coordinates[tetrahedron[2]] - &coordinates[tetrahedron[0]];
    let c = &coordinates[tetrahedron[3]] - &coordinates[tetrahedron[0]];
    triple_product(&a, &b, &c) / 6.0
}

fn triangle_skew<const D: usize>(
    a: &Coordinate<D>,
    b: &Coordinate<D>,
    c: &Coordinate<D>,
) -> Scalar {
    let l0 = (c - b).normalized();
    let l1 = (a - c).normalized();
    let l2 = (b - a).normalized();
    let minimum_angle = [
        (-(&l0 * &l1).value()).acos(),
        (-(&l1 * &l2).value()).acos(),
        (-(&l2 * &l0).value()).acos(),
    ]
    .into_iter()
    .fold(Scalar::INFINITY, Scalar::min);
    (EQUIANGLE - minimum_angle) / EQUIANGLE
}

fn maximum_edge_ratio<const D: usize, const E: usize>(
    edges: &[[usize; 2]; E],
    element: &[usize],
    coordinates: &Coordinates<D>,
) -> Scalar {
    let mut shortest = Quantity::new(Scalar::INFINITY);
    let mut longest = Quantity::default();
    for [a, b] in edges {
        let length = (&coordinates[element[*b]] - &coordinates[element[*a]]).norm();
        shortest = shortest.min(length);
        longest = longest.max(length);
    }
    if shortest > Quantity::default() {
        (longest / shortest).value()
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
        let normalizer: Scalar = edges.iter().map(|edge| edge.norm().value()).product();
        (corner_measure(&edges), normalizer)
    })
}

fn corner_measure<const D: usize, const K: usize>(edges: &[Coordinate<D>; K]) -> Scalar {
    if K == D {
        let matrix = from_fn(|i| from_fn(|j| edges[i][j]));
        TensorRank2::<K, Reference, Reference, _>::from(matrix).determinant()
    } else {
        let gram = from_fn(|i| from_fn(|j| &edges[i] * &edges[j]));
        TensorRank2::<K, Reference, Reference, _>::from(gram)
            .determinant()
            .max(0.0)
            .sqrt()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Kind {
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

pub(crate) fn minimum_scaled_jacobian<const D: usize>(
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
