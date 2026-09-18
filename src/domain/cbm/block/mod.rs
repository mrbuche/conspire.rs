pub mod node;
pub mod solid;
#[cfg(test)]
mod test;

use crate::{
    domain::{
        NodalReferenceCoordinates,
        block::{add_node_neighbors, element::Elements},
    },
    geometry::mesh::PrimitiveConnectivity,
};
use node::Node;
use std::fmt::{self, Debug, Formatter};

pub struct Block<C> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<3, 4>,
    nodes: Vec<Node>,
}

impl<C> Debug for Block<C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "Block {{ {} particles }}", self.nodes.len())
    }
}

impl<C>
    From<(
        C,
        PrimitiveConnectivity<3, 4>,
        &NodalReferenceCoordinates<3>,
    )> for Block<C>
{
    fn from(
        (constitutive_model, connectivity, reference_coordinates): (
            C,
            PrimitiveConnectivity<3, 4>,
            &NodalReferenceCoordinates<3>,
        ),
    ) -> Self {
        let nodes = Node::vec_from(&connectivity, reference_coordinates);
        Self {
            constitutive_model,
            connectivity,
            nodes,
        }
    }
}

impl<C> Elements for Block<C> {
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            self.connectivity.iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}
