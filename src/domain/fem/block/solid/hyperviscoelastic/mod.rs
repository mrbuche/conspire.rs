use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    fem::{
        NodalCoordinates,
        block::{
            Block, FiniteElementBlockError,
            element::solid::hyperviscoelastic::HyperviscoelasticFiniteElement,
            solid::elastic_hyperviscous::ElasticHyperviscousFiniteElementBlock,
        },
    },
    math::Scalar,
};

pub trait HyperviscoelasticFiniteElementBlock<C, F, const G: usize, const M: usize, const N: usize>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, M, N>,
    Self: ElasticHyperviscousFiniteElementBlock<C, F, G, M, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const M: usize, const N: usize>
    HyperviscoelasticFiniteElementBlock<C, F, G, M, N> for Block<C, F, N>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, M, N>,
    Self: ElasticHyperviscousFiniteElementBlock<C, F, G, M, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError> {
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
            Err(error) => Err(FiniteElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
