use crate::{
    constitutive::solid::viscoelastic::Viscoelastic,
    fem::{
        ElementModelError, NodalCoordinates, NodalVelocities,
        block::{
            Block, element::solid::viscoelastic::ViscoelasticFiniteElement, parallel_elements,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, viscoelastic::ViscoelasticElements},
    },
    mechanics::DeformationGradientRateList,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> Block<C, F, G, M, N, P>
where
    C: Viscoelastic + Sync,
    F: ViscoelasticFiniteElement<C, G, M, N, P> + Sync,
{
    pub fn deformation_gradient_rates(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
    ) -> Vec<DeformationGradientRateList<G>> {
        self.elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.deformation_gradient_rates(
                    &Self::element_coordinates(nodal_coordinates, nodes),
                    &Self::element_coordinates(nodal_velocities, nodes),
                )
            })
            .collect()
    }
}

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ViscoelasticElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Viscoelastic + Sync,
    F: ViscoelasticFiniteElement<C, G, M, N, P> + Sync,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
                &Self::element_coordinates(nodal_velocities, connectivity[index]),
            )
        }) {
            Ok(forces) => {
                forces.into_iter().flatten().zip(connectivity).for_each(
                    |(element_forces, nodes)| {
                        element_forces
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_force, &node)| nodal_forces[node] += nodal_force)
                    },
                );
                Ok(())
            }
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
    fn nodal_stiffnesses_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_velocities: &NodalVelocities<3>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
                &Self::element_coordinates(nodal_velocities, connectivity[index]),
            )
        }) {
            Ok(stiffnesses) => {
                stiffnesses
                    .into_iter()
                    .flatten()
                    .zip(connectivity)
                    .for_each(|(element_stiffness, nodes)| {
                        element_stiffness
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(object, &node_a)| {
                                object.into_iter().zip(nodes).for_each(
                                    |(nodal_stiffness, &node_b)| {
                                        nodal_stiffnesses[node_a][node_b] += nodal_stiffness
                                    },
                                )
                            })
                    });
                Ok(())
            }
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
