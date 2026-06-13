use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{ElementModelError, solid::elastic::ElasticElements},
    math::Tensor,
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{VirtualElementError, solid::elastic::ElasticVirtualElement},
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
        match self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), VirtualElementError>(())
            }) {
            Ok(()) => Ok(()),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<NodalStiffnessesSolid, ElementModelError> {
        let mut nodal_stiffnesses = NodalStiffnessesSolid::zero(nodal_coordinates.len());
        match self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                            })
                    });
                Ok::<(), VirtualElementError>(())
            }) {
            Ok(()) => Ok(nodal_stiffnesses),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
