use crate::{
    constitutive::{ConstitutiveError, solid::hyperviscoelastic::Hyperviscoelastic},
    domain::block::element::solid::hyperviscoelastic::HyperviscoelasticElement,
    math::{Quantity, Tensor},
    units::Energy,
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{SolidElement, elastic_hyperviscous::ElasticHyperviscousVirtualElement},
    },
};

pub trait HyperviscoelasticVirtualElement<C>
where
    C: Hyperviscoelastic,
    Self: ElasticHyperviscousVirtualElement<C> + HyperviscoelasticElement<C, 0>,
{
}

impl<T, C> HyperviscoelasticVirtualElement<C> for T
where
    C: Hyperviscoelastic,
    T: ElasticHyperviscousVirtualElement<C> + HyperviscoelasticElement<C, 0>,
{
}

impl<C, const P: usize> HyperviscoelasticElement<C, P> for Element
where
    C: Hyperviscoelastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
    ) -> Result<Quantity<Energy>, VirtualElementError> {
        let tetrahedra_energy = self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(nodal_coordinates).iter())
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
