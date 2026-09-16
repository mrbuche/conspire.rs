use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{
        ElementModelError,
        solid::{accumulate_nodal_forces, accumulate_nodal_stiffnesses, elastic::ElasticElements},
    },
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::solid::elastic::ElasticVirtualElement,
            solid::{NodalForcesSolid, NodalStiffnessesSolid},
        },
    },
};

impl<C, F> ElasticElements<3> for Block<C, F>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_forces: &mut NodalForcesSolid,
    ) -> Result<(), ElementModelError> {
        accumulate_nodal_forces(
            self.elements()
                .iter()
                .zip(self.elements_nodes())
                .map(|(element, nodes)| {
                    (
                        element
                            .nodal_forces(
                                self.constitutive_model(),
                                Self::element_coordinates(nodal_coordinates, nodes),
                            )
                            .map_err(|error| ElementModelError::upstream(error, self)),
                        nodes.as_slice(),
                    )
                }),
            nodal_forces,
        )
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_stiffnesses: &mut NodalStiffnessesSolid,
    ) -> Result<(), ElementModelError> {
        accumulate_nodal_stiffnesses(
            self.elements()
                .iter()
                .zip(self.elements_nodes())
                .map(|(element, nodes)| {
                    (
                        element
                            .nodal_stiffnesses(
                                self.constitutive_model(),
                                Self::element_coordinates(nodal_coordinates, nodes),
                            )
                            .map_err(|error| ElementModelError::upstream(error, self)),
                        nodes.as_slice(),
                    )
                }),
            nodal_stiffnesses,
        )
    }
}
