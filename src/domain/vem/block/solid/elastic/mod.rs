use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{ElementModelError, solid::elastic::ElasticItems},
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::solid::{
                ElementNodalForcesSolid, ElementNodalStiffnessesSolid,
                elastic::ElasticVirtualElement,
            },
        },
    },
};

impl<C, F, R> ElasticItems<3> for Block<C, F, R>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
{
    type Forces = ElementNodalForcesSolid;
    type Stiffnesses = ElementNodalStiffnessesSolid;
    fn number_of_items(&self) -> usize {
        self.elements().len()
    }
    fn nodes_of(&self, item: usize) -> &[usize] {
        &self.elements_nodes()[item]
    }
    fn forces_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<ElementNodalForcesSolid, ElementModelError> {
        self.elements()[item]
            .nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, &self.elements_nodes()[item]),
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn stiffnesses_of(
        &self,
        item: usize,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<ElementNodalStiffnessesSolid, ElementModelError> {
        self.elements()[item]
            .nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, &self.elements_nodes()[item]),
            )
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
