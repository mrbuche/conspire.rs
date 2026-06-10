#[cfg(test)]
mod test;

use super::Incidence;
use crate::{
    geometry::{
        Coordinate,
        mesh::{Mesh, Tessellation},
    },
    math::{Scalar, Tensor},
};
use std::mem::transmute_copy;

const PROBES: usize = 32;

impl<const D: usize> Mesh<D> {
    pub fn untangle(&mut self, iterations: usize, margin: Scalar, surface: Option<&Tessellation>) {
        let number_of_nodes = self.number_of_nodes();
        let neighbors = self.node_node_connectivity().to_vec();
        let incidence = Incidence::of(self);
        let constrained = surface.filter(|_| D == 3);
        let elements: Vec<&[usize]> = constrained
            .map(|surface| surface.mesh().connectivities().iter().flatten().collect())
            .unwrap_or_default();
        let mut boundary = vec![false; number_of_nodes];
        if constrained.is_some() {
            self.exterior_faces()
                .iter()
                .flatten()
                .for_each(|&node| boundary[node] = true);
        }
        let coordinates = self.coordinates.members_mut();
        for _ in 0..iterations {
            for node in 0..number_of_nodes {
                if neighbors[node].is_empty() {
                    continue;
                }
                let mut current = incidence.inversion(node, coordinates, margin);
                if current <= 0.0 {
                    continue;
                }
                let mut step = 0.5
                    * neighbors[node]
                        .iter()
                        .map(|&neighbor| (&coordinates[node] - &coordinates[neighbor]).norm())
                        .sum::<Scalar>()
                    / (neighbors[node].len() as Scalar);
                for _ in 0..PROBES {
                    let mut improved = false;
                    for axis in 0..D {
                        for sign in [-1.0, 1.0] {
                            let original = coordinates[node].clone();
                            coordinates[node][axis] += sign * step;
                            if boundary[node] {
                                coordinates[node] = project_to_surface(
                                    constrained.unwrap(),
                                    &elements,
                                    &coordinates[node],
                                );
                            }
                            let trial = incidence.inversion(node, coordinates, margin);
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

fn project_to_surface<const D: usize>(
    surface: &Tessellation,
    elements: &[&[usize]],
    point: &Coordinate<D>,
) -> Coordinate<D> {
    let query: &Coordinate<3> = unsafe { &*(point as *const Coordinate<D>).cast() };
    let projected = surface
        .bvh()
        .closest_point(query, surface.mesh().coordinates(), elements)
        .map_or_else(|| query.clone(), |(projected, _)| projected);
    unsafe { transmute_copy(&projected) }
}
