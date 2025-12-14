use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    fem::{
        NodalCoordinates,
        block::{
            ElementBlock, FiniteElementBlockError,
            element::HyperviscoelasticFiniteElement,
            solid::{
                SolidFiniteElementBlock,
                elastic_hyperviscous::ElasticHyperviscousFiniteElementBlock,
            },
        },
    },
    math::Scalar,
};

pub trait HyperviscoelasticFiniteElementBlock<C, F, const G: usize, const N: usize>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError>;
}

impl<C, F, const G: usize, const N: usize> HyperviscoelasticFiniteElementBlock<C, F, G, N>
    for ElementBlock<C, F, N>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticFiniteElement<C, G, N>,
    Self: ElasticHyperviscousFiniteElementBlock<C, F, G, N>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, FiniteElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.connectivity().iter())
            .map(|(element, element_connectivity)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &self.element_nodal_coordinates(element_connectivity, nodal_coordinates),
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
