mod hexahedron;
mod tetrahedron;

use crate::{
    geometry::{
        Coordinates,
        mesh::{Connectivity, Mesh},
    },
    math::{CrossProduct, Scalar, Tensor},
};
use std::array::from_fn;

pub trait Verdict {
    fn minimum_edge_ratios(&self) -> Vec<Vec<Scalar>>;
    fn minimum_jacobians(&self) -> Vec<Vec<Scalar>>;
    fn minimum_scaled_jacobians(&self) -> Vec<Vec<Scalar>>;
}

impl Verdict for Mesh<3> {
    fn minimum_edge_ratios(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| block_minimum_edge_ratios(block, coordinates))
            .collect()
    }
    fn minimum_jacobians(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| block_minimum_jacobians(block, coordinates))
            .collect()
    }
    fn minimum_scaled_jacobians(&self) -> Vec<Vec<Scalar>> {
        let coordinates = self.coordinates();
        self.iter()
            .map(|block| block_minimum_scaled_jacobians(block, coordinates))
            .collect()
    }
}

fn block_minimum_edge_ratios(block: &Connectivity, coordinates: &Coordinates<3>) -> Vec<Scalar> {
    match block {
        Connectivity::Hexahedral(elements) => elements
            .iter()
            .map(|element| hexahedron::minimum_edge_ratio(element, coordinates))
            .collect(),
        Connectivity::Tetrahedral(elements) => elements
            .iter()
            .map(|element| tetrahedron::minimum_edge_ratio(element, coordinates))
            .collect(),
        _ => todo!(),
    }
}

fn block_minimum_jacobians(block: &Connectivity, coordinates: &Coordinates<3>) -> Vec<Scalar> {
    match block {
        Connectivity::Hexahedral(elements) => elements
            .iter()
            .map(|element| hexahedron::minimum_jacobian(element, coordinates))
            .collect(),
        Connectivity::Tetrahedral(elements) => elements
            .iter()
            .map(|element| tetrahedron::minimum_jacobian(element, coordinates))
            .collect(),
        _ => todo!(),
    }
}

fn block_minimum_scaled_jacobians(
    block: &Connectivity,
    coordinates: &Coordinates<3>,
) -> Vec<Scalar> {
    match block {
        Connectivity::Hexahedral(elements) => elements
            .iter()
            .map(|element| hexahedron::minimum_scaled_jacobian(element, coordinates))
            .collect(),
        Connectivity::Tetrahedral(elements) => elements
            .iter()
            .map(|element| tetrahedron::minimum_scaled_jacobian(element, coordinates))
            .collect(),
        _ => todo!(),
    }
}

fn minimum_edge_ratio<const E: usize>(
    edges: &[[usize; 2]; E],
    element: &[usize],
    coordinates: &Coordinates<3>,
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

fn min_jacobian<const C: usize>(
    table: &[[usize; 3]; C],
    element: &[usize],
    coordinates: &Coordinates<3>,
) -> Scalar {
    corners(table, element, coordinates)
        .into_iter()
        .map(|(determinant, _)| determinant)
        .fold(Scalar::INFINITY, Scalar::min)
}

fn min_scaled_jacobian<const C: usize>(
    table: &[[usize; 3]; C],
    element: &[usize],
    coordinates: &Coordinates<3>,
    scale: Scalar,
) -> Scalar {
    corners(table, element, coordinates)
        .into_iter()
        .map(|(determinant, normalizer)| {
            if normalizer > 0.0 {
                (scale * determinant / normalizer).clamp(-1.0, 1.0)
            } else {
                0.0
            }
        })
        .fold(Scalar::INFINITY, Scalar::min)
}

fn corners<const C: usize>(
    table: &[[usize; 3]; C],
    element: &[usize],
    coordinates: &Coordinates<3>,
) -> [(Scalar, Scalar); C] {
    from_fn(|corner| {
        let [a, b, c] = table[corner];
        let origin = &coordinates[element[corner]];
        let edge_a = &coordinates[element[a]] - origin;
        let edge_b = &coordinates[element[b]] - origin;
        let edge_c = &coordinates[element[c]] - origin;
        let determinant = &edge_a.cross(&edge_b) * &edge_c;
        let normalizer = edge_a.norm() * edge_b.norm() * edge_c.norm();
        (determinant, normalizer)
    })
}