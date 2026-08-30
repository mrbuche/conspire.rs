use crate::math::Quantity;
use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic::Hyperelastic},
    fem::block::element::solid::hyperelastic::HyperelasticFiniteElement,
    math::Tensor,
    units::Energy,
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{SolidVirtualElement, elastic::ElasticVirtualElement},
    },
};

pub trait HyperelasticVirtualElement<C>
where
    C: Hyperelastic,
    Self: ElasticVirtualElement<C>,
{
    fn helmholtz_free_energy<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<Quantity<Energy>, VirtualElementError>;
}

impl<C> HyperelasticVirtualElement<C> for Element
where
    C: Hyperelastic,
    Self: ElasticVirtualElement<C>,
{
    fn helmholtz_free_energy<'a>(
        &'a self,
        constitutive_model: &'a C,
        nodal_coordinates: ElementNodalCoordinates<'a>,
    ) -> Result<Quantity<Energy>, VirtualElementError> {
        let tetrahedra_energy = self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(&nodal_coordinates).iter())
            .map(|(tetrahedron, tetrahedron_coordinates)| {
                tetrahedron.helmholtz_free_energy(constitutive_model, tetrahedron_coordinates)
            })
            .sum::<Result<Quantity<Energy>, _>>()
            .map_err(|error| self.upstream(error))?;
        let polyhedron_energy = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(self.integration_weights())
            .map(|(deformation_gradient, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.helmholtz_free_energy_density(deformation_gradient)?
                        * integration_weight,
                )
            })
            .sum::<Result<Quantity<Energy>, _>>()
            .map_err(|error| self.upstream(error))?;
        Ok(polyhedron_energy * (1.0 - self.stabilization())
            + tetrahedra_energy * self.stabilization())
    }
}
