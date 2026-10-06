use crate::{
    constitutive::solid::elastic::Elastic,
    domain::{
        ElementModelError,
        feti::block::element::{DecomposableElements, ElementSystem, ElementSystems, positions},
    },
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{VirtualElementError, solid::elastic::ElasticVirtualElement},
        },
    },
};

impl<C, F, R> DecomposableElements for Block<C, F, R>
where
    C: Elastic,
    F: ElasticVirtualElement<C>,
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
                    nodes.to_vec(),
                    |a, i| forces[a][i].value(),
                    |a, b, i, j| stiffnesses[a][b][i][j].value(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| ElementModelError::upstream(error, self))?;
        Ok(ElementSystems {
            positions: positions(nodal_coordinates),
            elements,
        })
    }
}
