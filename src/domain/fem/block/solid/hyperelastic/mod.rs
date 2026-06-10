use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    fem::{
        FiniteElementModelError, NodalCoordinates,
        block::{Block, element::solid::hyperelastic::HyperelasticFiniteElement},
        solid::{elastic::ElasticFiniteElements, hyperelastic::HyperelasticFiniteElements},
    },
    math::Scalar,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperelasticFiniteElements for Block<C, F, G, M, N, P>
where
    C: Hyperelastic,
    F: HyperelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticFiniteElements,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementModelError> {
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
            Err(error) => Err(FiniteElementModelError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
