use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    domain::{ElementModelError, solid::viscoelastic::ViscoelasticElements},
    vem::{
        NodalCoordinates, NodalVelocities,
        block::{
            Block,
            element::{VirtualElementError, solid::viscoelastic::ViscoelasticVirtualElement},
            solid::{NodalDampingsSolid, NodalForcesSolid},
        },
    },
};

impl<C, F> ViscoelasticElements<3> for Block<C, F>
where
    C: Viscoelastic,
    F: ViscoelasticVirtualElement<C>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
        nodal_forces: &mut NodalForcesSolid,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_forces(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &Self::element_coordinates(nodal_velocities, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force);
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_velocities: &NodalVelocities,
        nodal_stiffnesses: &mut NodalDampingsSolid,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                        &Self::element_coordinates(nodal_velocities, nodes),
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
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
