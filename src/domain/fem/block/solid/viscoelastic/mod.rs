use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    fem::{
        ElementModelError, NodalCoordinates, NodalVelocities,
        block::{
            Block,
            element::{FiniteElementError, solid::viscoelastic::ViscoelasticFiniteElement},
        },
        solid::{NodalDampingsSolid, NodalForcesSolid, viscoelastic::ViscoelasticElements},
    },
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ViscoelasticElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Viscoelastic,
    F: ViscoelasticFiniteElement<C, G, M, N, P>,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
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
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_stiffnesses: &mut NodalDampingsSolid<3>,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.connectivity())
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
                Ok::<(), FiniteElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
