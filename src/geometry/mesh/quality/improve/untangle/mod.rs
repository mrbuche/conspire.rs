#[cfg(test)]
mod test;

use super::Incidence;
use crate::{
    geometry::mesh::Mesh,
    math::{Scalar, Tensor},
};

const PROBES: usize = 32;

impl<const D: usize> Mesh<D> {
    /// Derivative-free untangling: each tangled node (incident minimum scaled
    /// Jacobian <= 0) is moved by a shrinking compass search that maximizes that
    /// minimum. Only strictly improving probes are kept, so the global minimum
    /// scaled Jacobian never decreases and no new tangles are created.
    pub fn untangle(&mut self, iterations: usize) {
        let number_of_nodes = self.number_of_nodes();
        let neighbors = self.node_node_connectivity().to_vec();
        let incidence = Incidence::of(self);
        let coordinates = self.coordinates.members_mut();
        for _ in 0..iterations {
            for node in 0..number_of_nodes {
                if neighbors[node].is_empty() {
                    continue;
                }
                let mut current = incidence.minimum_jacobian(node, coordinates);
                if current > 0.0 {
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
                            let original = coordinates[node][axis];
                            coordinates[node][axis] = original + sign * step;
                            let trial = incidence.minimum_jacobian(node, coordinates);
                            if trial > current {
                                current = trial;
                                improved = true;
                            } else {
                                coordinates[node][axis] = original;
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
