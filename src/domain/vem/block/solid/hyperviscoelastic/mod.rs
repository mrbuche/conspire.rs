use crate::{
    constitutive::solid::hyperviscoelastic::Hyperviscoelastic,
    domain::{
        ElementModelError,
        solid::{
            elastic_hyperviscous::ElasticHyperviscousElements,
            hyperviscoelastic::HyperviscoelasticElements,
        },
    },
    math::Quantity,
    units::Energy,
    vem::{
        NodalCoordinates,
        block::{
            Block,
            element::{
                VirtualElementError, solid::hyperviscoelastic::HyperviscoelasticVirtualElement,
            },
        },
    },
};

impl<C, F> HyperviscoelasticElements<3> for Block<C, F>
where
    C: Hyperviscoelastic,
    F: HyperviscoelasticVirtualElement<C>,
    Self: ElasticHyperviscousElements<3>,
{
    fn helmholtz_free_energy(
        &self,
        nodal_coordinates: &NodalCoordinates,
    ) -> Result<Quantity<Energy>, ElementModelError> {
        self.elements()
            .iter()
            .zip(self.elements_nodes())
            .map(|(element, nodes)| {
                element.helmholtz_free_energy(
                    self.constitutive_model(),
                    &Self::element_coordinates(nodal_coordinates, nodes),
                )
            })
            .sum::<Result<_, VirtualElementError>>()
            .map_err(|error| ElementModelError::upstream(error, self))
    }
}
