pub mod density;
mod diffusion;
#[cfg(test)]
mod electrical;
pub mod mass;
pub mod node;
pub mod solid;
#[cfg(test)]
mod test;
pub mod thermal;
pub mod time_scale;
mod volume;

use crate::{
    domain::{NodalReferenceCoordinates, block::element::Items},
    geometry::mesh::PrimitiveConnectivity,
};
use core::fmt::{self, Debug, Formatter};
use density::NoDensity;
use node::{Node, Weighting};

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
        (
            constitutive_model,
            connectivity,
            reference_coordinates,
            Weighting::default(),
        )
            .into()
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

impl<C, R> Items for Block<C, R> {
    fn number_of_items(&self) -> usize {
        self.nodes.len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        self.nodes[item].neighbors()
    }
}
