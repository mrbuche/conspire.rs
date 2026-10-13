#[cfg(test)]
mod test;

use super::{
    super::metrics::{Kind, hexahedron::CORNERS},
    Incidence,
};
use crate::{
    geometry::{Coordinate, Coordinates, mesh::Mesh},
    math::{Quantity, Scalar, Tensor},
};

const EPSILON: Scalar = 1.0e-12;
const PROBES: usize = 32;

#[derive(Clone, Copy, Debug)]
pub enum CornerEnergy {
    Cosine,
    Signed,
}

impl Mesh<3> {
    pub fn corner_energy_smooth(&mut self, iterations: usize, energy: CornerEnergy) {
        let mut fixed = vec![false; self.number_of_nodes()];
        self.exterior_faces()
            .iter()
            .flatten()
            .for_each(|&node| fixed[node] = true);
        self.corner_energy_smooth_fixed(iterations, energy, &fixed)
    }
    pub(crate) fn corner_energy_smooth_fixed(
        &mut self,
        iterations: usize,
        energy: CornerEnergy,
        fixed: &[bool],
    ) {
        let neighbors = self.node_node_connectivity().to_vec();
        let incidence = Incidence::of(self);
        assert!(
            incidence
                .element_kinds
                .iter()
                .all(|kind| matches!(kind, Kind::Hexahedron)),
            "corner energy smoothing supports only hexahedra"
        );
        let coordinates = self.coordinates.members_mut();
        for _ in 0..iterations {
            for node in 0..neighbors.len() {
                if fixed[node] || neighbors[node].is_empty() {
                    continue;
                }
                let mut current = incidence.corner_energy(node, coordinates, energy);
                let mut step = 0.5
                    * neighbors[node]
                        .iter()
                        .map(|&neighbor| {
                            (&coordinates[node] - &coordinates[neighbor]).norm().value()
                        })
                        .sum::<Scalar>()
                    / (neighbors[node].len() as Scalar);
                for _ in 0..PROBES {
                    let mut improved = false;
                    for axis in 0..3 {
                        for sign in [-1.0, 1.0] {
                            let original = coordinates[node].clone();
                            coordinates[node][axis] += Quantity::new(sign * step);
                            let trial = incidence.corner_energy(node, coordinates, energy);
                            if trial < current {
                                current = trial;
                                improved = true;
                            } else {
                                coordinates[node] = original;
                            }
                        }
                    }
                    if !improved {
                        step *= 0.5;
                    }
                }
            }
        }
    }
}

impl Incidence {
    fn corner_energy(
        &self,
        node: usize,
        coordinates: &Coordinates<3>,
        energy: CornerEnergy,
    ) -> Scalar {
        self.node_elements[node]
            .iter()
            .map(|&element| element_energy(&self.element_nodes[element], coordinates, energy))
            .sum()
    }
}

pub(crate) fn element_energy(
    element: &[usize],
    coordinates: &Coordinates<3>,
    energy: CornerEnergy,
) -> Scalar {
    CORNERS
        .iter()
        .map(|(origin, neighbors)| {
            let origin = &coordinates[element[*origin]];
            let edges = neighbors.map(|neighbor| unit(&coordinates[element[neighbor]] - origin));
            match edges {
                [Some(a), Some(b), Some(c)] => corner_energy(&[a, b, c], energy),
                _ => Scalar::INFINITY,
            }
        })
        .sum()
}

fn corner_energy(edges: &[[Scalar; 3]; 3], energy: CornerEnergy) -> Scalar {
    match energy {
        CornerEnergy::Cosine => [(0, 1), (0, 2), (1, 2)]
            .into_iter()
            .map(|(i, j)| dot(&edges[i], &edges[j]).powi(2))
            .sum(),
        CornerEnergy::Signed => [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
            .into_iter()
            .map(|(i, j, k)| {
                let normal = cross(&edges[i], &edges[j]);
                1.0 - dot(&edges[k], &normal) / dot(&normal, &normal).sqrt().max(EPSILON)
            })
            .sum(),
    }
}

fn unit(edge: Coordinate<3>) -> Option<[Scalar; 3]> {
    let vector = [edge[0].value(), edge[1].value(), edge[2].value()];
    let length = dot(&vector, &vector).sqrt();
    (length > 0.0).then(|| vector.map(|component| component / length))
}

fn dot(a: &[Scalar; 3], b: &[Scalar; 3]) -> Scalar {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: &[Scalar; 3], b: &[Scalar; 3]) -> [Scalar; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
