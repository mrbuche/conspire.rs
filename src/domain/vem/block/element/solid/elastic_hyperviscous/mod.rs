use crate::{
    constitutive::{ConstitutiveError, solid::elastic_hyperviscous::ElasticHyperviscous},
    domain::block::element::solid::elastic_hyperviscous::ElasticHyperviscousElement,
    math::{Quantity, Tensor},
    units::Power,
    vem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, VirtualElement,
        VirtualElementError,
        solid::{SolidElement, viscoelastic::ViscoelasticVirtualElement},
    },
};

pub trait ElasticHyperviscousVirtualElement<C>
where
    C: ElasticHyperviscous,
    Self: ViscoelasticVirtualElement<C> + ElasticHyperviscousElement<C, 0>,
{
}

impl<T, C> ElasticHyperviscousVirtualElement<C> for T
where
    C: ElasticHyperviscous,
    T: ViscoelasticVirtualElement<C> + ElasticHyperviscousElement<C, 0>,
{
}

impl<C, const P: usize> ElasticHyperviscousElement<C, P> for Element
where
    C: ElasticHyperviscous,
{
    fn viscous_dissipation(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> Result<Quantity<Power>, VirtualElementError> {
        let tetrahedra_dissipation = self
            .tetrahedra()
            .iter()
            .zip(
                self.tetrahedra_coordinates(nodal_coordinates)
                    .iter()
                    .zip(self.tetrahedra_coordinates(nodal_velocities).iter()),
            )
            .map(
                |(tetrahedron, (tetrahedron_coordinates, tetrahedron_velocities))| {
                    tetrahedron.viscous_dissipation(
                        constitutive_model,
                        tetrahedron_coordinates,
                        tetrahedron_velocities,
                    )
                },
            )
            .sum::<Result<Quantity<Power>, _>>()
            .map_err(|error| self.upstream(error))?;
        let polyhedron_dissipation = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model
                            .viscous_dissipation(deformation_gradient, deformation_gradient_rate)?
                            * integration_weight,
                    )
                },
            )
            .sum::<Result<Quantity<Power>, _>>()
            .map_err(|error| self.upstream(error))?;
        Ok(polyhedron_dissipation * (1.0 - self.stabilization())
            + tetrahedra_dissipation * self.stabilization())
    }
    fn dissipation_potential(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        nodal_velocities: &ElementNodalVelocities,
    ) -> Result<Quantity<Power>, VirtualElementError> {
        let tetrahedra_potential = self
            .tetrahedra()
            .iter()
            .zip(
                self.tetrahedra_coordinates(nodal_coordinates)
                    .iter()
                    .zip(self.tetrahedra_coordinates(nodal_velocities).iter()),
            )
            .map(
                |(tetrahedron, (tetrahedron_coordinates, tetrahedron_velocities))| {
                    tetrahedron.dissipation_potential(
                        constitutive_model,
                        tetrahedron_coordinates,
                        tetrahedron_velocities,
                    )
                },
            )
            .sum::<Result<Quantity<Power>, _>>()
            .map_err(|error| self.upstream(error))?;
        let polyhedron_potential = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(
                self.deformation_gradient_rates(nodal_coordinates, nodal_velocities)
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(deformation_gradient, (deformation_gradient_rate, integration_weight))| {
                    Ok::<_, ConstitutiveError>(
                        constitutive_model.dissipation_potential(
                            deformation_gradient,
                            deformation_gradient_rate,
                        )? * integration_weight,
                    )
                },
            )
            .sum::<Result<Quantity<Power>, _>>()
            .map_err(|error| self.upstream(error))?;
        Ok(polyhedron_potential * (1.0 - self.stabilization())
            + tetrahedra_potential * self.stabilization())
    }
}
