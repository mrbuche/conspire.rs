use super::super::{Block, node::solid::elastic::ElasticElement};
use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{ElementModelError, NodalCoordinates, solid::elastic::ElasticItems},
    math::{Current, TensorRank1, TensorRank2},
    units::{Force, ForcePerLength},
};

pub use crate::domain::solid::elastic::ElasticElements;

type NodalForce = TensorRank1<3, Current, Force>;
type NodalStiffness = TensorRank2<3, Current, Current, ForcePerLength>;

impl<C, R> ElasticItems<3> for Block<C, R>
where
    C: Elastic,
{
    type Forces = Vec<NodalForce>;
    type Stiffnesses = Vec<Vec<NodalStiffness>>;
    fn number_of_items(&self) -> usize {
        self.nodes.len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        self.nodes[item].neighbors()
    }
    fn forces_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Vec<NodalForce>, ElementModelError> {
        self.nodes[item]
            .nodal_forces(&self.constitutive_model, nodal_coordinates)
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn stiffnesses_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Vec<Vec<NodalStiffness>>, ElementModelError> {
        self.nodes[item]
            .nodal_stiffnesses(&self.constitutive_model, nodal_coordinates)
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
