use crate::{
    constitutive::{ConstitutiveError, solid::hyperelastic_plastic::HyperelasticPlastic},
    domain::block::element::solid::{
        hyperelastic_plastic::HyperelasticPlasticElement, plastic::PlasticStateVariables,
    },
    math::{Quantity, Tensor},
    units::Energy,
    vem::block::element::{
        Element, ElementNodalCoordinates, VirtualElement, VirtualElementError,
        solid::{SolidElement, elastic_plastic::ElasticPlasticVirtualElement},
    },
};

pub trait HyperelasticPlasticVirtualElement<C>
where
    C: HyperelasticPlastic,
    Self: ElasticPlasticVirtualElement<C> + HyperelasticPlasticElement<C, 1>,
{
}

impl<T, C> HyperelasticPlasticVirtualElement<C> for T
where
    C: HyperelasticPlastic,
    T: ElasticPlasticVirtualElement<C> + HyperelasticPlasticElement<C, 1>,
{
}

impl<C> HyperelasticPlasticElement<C, 1> for Element
where
    C: HyperelasticPlastic,
{
    fn helmholtz_free_energy(
        &self,
        constitutive_model: &C,
        nodal_coordinates: &ElementNodalCoordinates,
        state_variables: &PlasticStateVariables<1>,
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
                    Ok(constitutive_model
                        .helmholtz_free_energy_density(deformation_gradient, &state_variable.0)?
                        * integration_weight)
                },
            )
            .sum::<Result<Quantity<Energy>, ConstitutiveError>>()
            .map_err(|error| self.upstream(error))?;
        Ok(polyhedron_energy * (1.0 - self.stabilization())
            + tetrahedra_energy * self.stabilization())
    }
}
