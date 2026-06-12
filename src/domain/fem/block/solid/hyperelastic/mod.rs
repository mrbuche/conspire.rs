use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{Block, element::solid::hyperelastic::HyperelasticFiniteElement},
        solid::{elastic::ElasticElements, hyperelastic::HyperelasticElements},
    },
    math::Scalar,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize> HyperelasticElements<3>
    for Block<C, F, G, M, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticElements<3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates<3>,
    ) -> Result<Scalar, ElementModelError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
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
