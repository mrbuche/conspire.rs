use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    fem::{
        FiniteElementModelError, NodalCoordinates,
        block::{Block, element::solid::hyperviscoelastic::HyperviscoelasticFiniteElement},
        solid::{
            elastic_hyperviscous::ElasticHyperviscousFiniteElements,
            hyperviscoelastic::HyperviscoelasticFiniteElements,
        },
    },
    math::Scalar,
};

impl<C, F, const G: usize, const M: usize, const N: usize, const P: usize>
    HyperviscoelasticFiniteElements for Block<C, F, G, M, N, P>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, M, N, P>,
    Self: ElasticHyperviscousFiniteElements,
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
