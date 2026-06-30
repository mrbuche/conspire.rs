use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    fem::{
        ElementModelError, NodalCoordinates,
        block::{Block, element::solid::hyperviscoelastic::HyperviscoelasticFiniteElement},
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements,
            hyperviscoelastic::HyperviscoelasticElements,
        },
    },
    math::Scalar,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperviscoelasticElements<3> for Block<C, F, G, M, N, P>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticHyperviscousElements<3>,
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
