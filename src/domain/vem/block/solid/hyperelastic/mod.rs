use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    domain::{
        ElementModelError,
        block::feti::element_systems::{DecomposableElements, ElementSystem, ElementSystems},
        solid::hyperelastic::HyperelasticElements,
    },
    math::{HessianAccumulate, Quantity, Tensor},
    units::Energy,
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{VirtualElementError, solid::hyperelastic::HyperelasticVirtualElement},
            solid::NodalStiffnessesSolidSymmetric,
        },
    },
};

impl<C, F> HyperelasticElements<3> for Block<C, F>
where
    C: Hyperelastic,
    F: HyperelasticVirtualElement<C>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
    fn nodal_stiffnesses_symmetric_into(
        &self,
        nodal_coordinates: &NodalCoordinates,
        nodal_stiffnesses: &mut NodalStiffnessesSolidSymmetric,
    ) -> Result<(), ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .try_for_each(|(element, nodes)| {
                element
                    .nodal_stiffnesses(
                        self.constitutive_model(),
                        &Self::element_coordinates(nodal_coordinates, nodes),
                    )?
                    .into_iter()
                    .zip(nodes)
                    .for_each(|(object, &node_a)| {
                        object
                            .into_iter()
                            .zip(nodes)
                            .for_each(|(nodal_stiffness, &node_b)| {
                                if node_a <= node_b {
                                    nodal_stiffnesses.accumulate(node_a, node_b, nodal_stiffness)
                                }
                            })
                    });
                Ok::<(), VirtualElementError>(())
            })
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}

impl<C, F> DecomposableElements for Block<C, F>
where
    C: Hyperelastic,
    F: HyperelasticVirtualElement<C>,
{
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<ElementSystems, ElementModelError> {
        let elements = self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                let coordinates = Self::element_coordinates(nodal_coordinates, nodes);
                let forces = element.nodal_forces(self.constitutive_model(), &coordinates)?;
                let stiffnesses =
                    element.nodal_stiffnesses(self.constitutive_model(), &coordinates)?;
                Ok::<_, VirtualElementError>(ElementSystem::pack(
                    nodes.clone(),
                    |a, i| forces[a][i].value(),
                    |a, b, i, j| stiffnesses[a][b][i][j].value(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ElementModelError::upstream(error, self))?;
        Ok(ElementSystems {
            number_of_nodes: nodal_coordinates.len(),
            elements,
        })
    }
}
