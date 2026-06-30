use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{ElementModelError, solid::hyperelastic::HyperelasticElements},
    math::Scalar,
    vem::{
        NodalCoordinates,
        block::{Block, element::solid::hyperelastic::HyperelasticVirtualElement},
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
    ) -> Result<Scalar, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(ElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
