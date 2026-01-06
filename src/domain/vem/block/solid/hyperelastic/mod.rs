use crate::{
    constitutive::solid::hyperelastic::Hyperelastic,
    math::Scalar,
    vem::{
        NodalCoordinates,
        block::{
            Block, VirtualElementBlockError,
            element::solid::hyperelastic::HyperelasticVirtualElement,
            solid::SolidVirtualElementBlock,
        },
    },
};

pub trait HyperelasticVirtualElementBlock<C, F>
where
    C: Hyperelastic,
    F: HyperelasticVirtualElement<C>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, VirtualElementBlockError>;
}

impl<C, F> HyperelasticVirtualElementBlock<C, F> for Block<C, F>
where
    C: Hyperelastic,
    F: HyperelasticVirtualElement<C>,
    Self: SolidVirtualElementBlock<C, F>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Scalar, VirtualElementBlockError> {
        match self
            .elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    self.element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum()
        {
            Ok(helmholtz_free_energy) => Ok(helmholtz_free_energy),
            Err(error) => Err(VirtualElementBlockError::Upstream(
                format!("{error}"),
                format!("{self:?}"),
            )),
        }
    }
}
