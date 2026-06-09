mod smart_laplace;
mod untangle;

use super::metrics::{self, Kind};
use crate::{
    geometry::{Coordinates, mesh::Mesh},
    math::Scalar,
};

struct Incidence {
    node_elements: Vec<Vec<usize>>,
    element_nodes: Vec<Vec<usize>>,
    element_kinds: Vec<Kind>,
}

impl Incidence {
    fn of<const D: usize>(mesh: &Mesh<D>) -> Self {
        Self {
            node_elements: mesh.node_element_connectivity().to_vec(),
            element_nodes: mesh
                .iter()
                .flat_map(|block| block.iter().map(<[usize]>::to_vec))
                .collect(),
            element_kinds: mesh
                .iter()
                .flat_map(|block| {
                    let kind = Kind::of(block).expect("unsupported element type");
                    block.iter().map(move |_| kind)
                })
                .collect(),
        }
    }
    fn minimum_jacobian<const D: usize>(
        &self,
        node: usize,
        coordinates: &Coordinates<D>,
    ) -> Scalar {
        self.minimum(node, coordinates, metrics::minimum_jacobian)
    }
    fn minimum_scaled_jacobian<const D: usize>(
        &self,
        node: usize,
        coordinates: &Coordinates<D>,
    ) -> Scalar {
        self.minimum(node, coordinates, metrics::minimum_scaled_jacobian)
    }
    fn minimum<const D: usize>(
        &self,
        node: usize,
        coordinates: &Coordinates<D>,
        metric: impl Fn(Kind, &[usize], &Coordinates<D>) -> Scalar,
    ) -> Scalar {
        self.node_elements[node]
            .iter()
            .map(|&element| {
                metric(
                    self.element_kinds[element],
                    &self.element_nodes[element],
                    coordinates,
                )
            })
            .fold(Scalar::INFINITY, Scalar::min)
    }
}
