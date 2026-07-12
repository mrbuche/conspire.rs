use crate::{
    constitutive::solid::elastic::Elastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{planar::PlanarElasticFiniteElement, solid::elastic::ElasticFiniteElement},
            parallel_elements,
        },
        solid::{NodalForcesSolid, NodalStiffnessesSolid, elastic::ElasticElements},
    },
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> ElasticElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Elastic + Sync,
    F: ElasticFiniteElement<C, G, M, N, P> + Sync,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
        nodal_forces: &mut NodalForcesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
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
        nodal_stiffnesses: &mut NodalStiffnessesSolid<3>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
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

impl<C, F, const G: usize, const N: usize, const P: usize> ElasticElements<2>
    for Block<C, F, G, 2, N, P>
where
    C: Elastic + Sync,
    F: PlanarElasticFiniteElement<C, G, N, P> + Sync,
{
    fn nodal_forces_into(
        &self,
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_forces: &mut NodalForcesSolid<2>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_forces(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
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
        nodal_coordinates: &NodalCoordinates<2>,
        nodal_stiffnesses: &mut NodalStiffnessesSolid<2>,
    ) -> Result<(), ElementModelError> {
        let elements = self.elements();
        let connectivity: Vec<&[usize; N]> = self.connectivity().iter().collect();
        match parallel_elements(elements.len(), |index| {
            elements[index].nodal_stiffnesses(
                self.constitutive_model(),
                &Self::element_coordinates(nodal_coordinates, connectivity[index]),
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
