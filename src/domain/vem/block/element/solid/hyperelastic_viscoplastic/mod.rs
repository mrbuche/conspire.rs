use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_viscoplastic::HyperelasticViscoplastic},
    domain::block::element::solid::{
        hyperelastic_viscoplastic::HyperelasticViscoplasticElement,
        viscoplastic::ViscoplasticStateVariables,
    },
    math::{Differentiable, Quantity, Tensor},
    units::Energy,
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{SolidElement, elastic_viscoplastic::ElasticViscoplasticVirtualElement},
    },
};

pub trait HyperelasticViscoplasticVirtualElement<C, Y>
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
    Self: ElasticViscoplasticVirtualElement<C, Y> + HyperelasticViscoplasticElement<C, 1, Y>,
{
}

impl<T, C, Y> HyperelasticViscoplasticVirtualElement<C, Y> for T
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
    T: ElasticViscoplasticVirtualElement<C, Y> + HyperelasticViscoplasticElement<C, 1, Y>,
{
}

impl<C, Y> HyperelasticViscoplasticElement<C, 1, Y> for Element
where
    C: HyperelasticViscoplastic<Y>,
    Y: Differentiable + Tensor,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        state_variables: &ViscoplasticStateVariables<1, Y>,
    ) -> Result<Quantity<Energy>, VirtualElementError> {
        let tetrahedra_energy = self
            .tetrahedra()
            .iter()
            .zip(self.tetrahedra_coordinates(nodal_coordinates).iter())
            .map(|(tetrahedron, tetrahedron_coordinates)| {
                tetrahedron.helmholtz_free_energy(
                    constitutive_model,
                    tetrahedron_coordinates,
                    state_variables,
                )
            })
            .sum::<Result<Quantity<Energy>, _>>()
            .map_err(|error| self.upstream(error))?;
        let polyhedron_energy = self
            .deformation_gradients(nodal_coordinates)
            .iter()
            .zip(state_variables)
            .zip(self.integration_weights())
            .map(
                |((deformation_gradient, state_variable), integration_weight)| {
                    let (deformation_gradient_p, _) = state_variable.into();
                    Ok::<_, ConstitutiveError>(
                        constitutive_model.helmholtz_free_energy_density(
                            deformation_gradient,
                            deformation_gradient_p,
                        )? * integration_weight,
                    )
                },
            )
            .sum::<Result<Quantity<Energy>, _>>()
            .map_err(|error| self.upstream(error))?;
        Ok(polyhedron_energy * (1.0 - self.stabilization())
            + tetrahedra_energy * self.stabilization())
    }
}
