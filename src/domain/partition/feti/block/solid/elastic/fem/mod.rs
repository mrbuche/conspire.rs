use crate::{
    constitutive::solid::elastic::Elastic,
    domain::feti::block::element::{
        DecomposableElements, ElementSystem, ElementSystems, positions,
    },
    fem::{
        ElementModelError, NodalCoordinates,
        block::{
            Block,
            element::{FiniteElementError, solid::elastic::ElasticFiniteElement},
        },
    },
};

impl<C, F, const G: usize, const N: usize, const P: usize> DecomposableElements
    for Block<C, F, G, 3, N, P>
where
    C: Elastic,
    F: ElasticFiniteElement<C, G, 3, N, P>,
{
    fn element_systems(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<ElementSystems, ElementModelError> {
        let elements = self
            .connectivity()
            .iter()
            .zip(self.elements())
            .map(|(nodes, element)| {
                let coordinates = Self::element_coordinates(nodal_coordinates, nodes);
                let forces = element.nodal_forces(self.constitutive_model(), &coordinates)?;
                let stiffnesses =
                    element.nodal_stiffnesses(self.constitutive_model(), &coordinates)?;
                Ok::<_, FiniteElementError>(ElementSystem::pack(
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

#[cfg(test)]
mod test;
