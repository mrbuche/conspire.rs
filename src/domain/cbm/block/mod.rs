pub mod density;
pub mod mass;
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
use density::NoDensity;
use node::{Node, Weighting};
use std::fmt::{self, Debug, Formatter};

pub struct Block<C, R = NoDensity> {
    constitutive_model: C,
    connectivity: PrimitiveConnectivity<3, 4>,
    nodes: Vec<Node>,
    density: R,
}

impl<C, R> Debug for Block<C, R> {
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
        Self::from((
            constitutive_model,
            connectivity,
            reference_coordinates,
            Weighting::default(),
        ))
    }
}

impl<C>
    From<(
        C,
        PrimitiveConnectivity<3, 4>,
        &NodalReferenceCoordinates<3>,
        Weighting,
    )> for Block<C>
{
    fn from(
        (constitutive_model, connectivity, reference_coordinates, weighting): (
            C,
            PrimitiveConnectivity<3, 4>,
            &NodalReferenceCoordinates<3>,
            Weighting,
        ),
    ) -> Self {
        let nodes = Node::vec_from(&connectivity, reference_coordinates, weighting);
        Self {
            constitutive_model,
            connectivity,
            nodes,
            density: NoDensity,
        }
    }
}

impl<C, R> Elements for Block<C, R> {
    fn node_neighbors(&self, neighbors: &mut [Vec<usize>]) {
        add_node_neighbors(
            self.connectivity.iter().map(|nodes| nodes.as_slice()),
            neighbors,
        )
    }
}
