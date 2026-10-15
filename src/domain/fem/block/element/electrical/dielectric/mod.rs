#[cfg(test)]
pub mod test;

use crate::{
    constitutive::{ConstitutiveError, electrical::dielectric::Dielectric},
    fem::block::element::{
        Element, FiniteElement, FiniteElementError,
        electrical::{ElectricalFiniteElement, ElementNodalPotentials},
    },
    math::{Quantity, Tensor, TensorList},
    mechanics::{ElectricDisplacements, ElectricTangents},
    units::{Capacitance, Charge, Energy},
};

pub type ElementNodalForcesElectrical<const D: usize> = TensorList<Quantity<Charge>, D>;
pub type ElementNodalStiffnessesElectrical<const D: usize> =
    TensorList<ElementNodalForcesElectricalPerPotential<D>, D>;
pub type ElementNodalForcesElectricalPerPotential<const D: usize> =
    TensorList<Quantity<Capacitance>, D>;

pub trait DielectricFiniteElement<C, const G: usize, const M: usize, const N: usize, const P: usize>
where
    C: Dielectric,
    Self: ElectricalFiniteElement<G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> Result<Quantity<Energy>, FiniteElementError>;
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> Result<ElementNodalForcesElectrical<N>, FiniteElementError>;
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> Result<ElementNodalStiffnessesElectrical<N>, FiniteElementError>;
}

impl<C, const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    DielectricFiniteElement<C, G, M, N, P> for Element<3, G, N, O>
where
    C: Dielectric,
    Self: ElectricalFiniteElement<G, M, N, P>,
{
    fn potential(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> Result<Quantity<Energy>, FiniteElementError> {
        self.electric_fields(nodal_potentials)
            .iter()
            .zip(self.integration_weights())
            .map(|(electric_field, integration_weight)| {
                Ok::<_, ConstitutiveError>(
                    constitutive_model.potential(electric_field)? * integration_weight,
                )
            })
            .sum::<Result<_, ConstitutiveError>>()
            .map_err(|error| FiniteElementError::upstream(error, self))
    }
    fn nodal_forces(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> Result<ElementNodalForcesElectrical<N>, FiniteElementError> {
        let electric_displacements = self
            .electric_fields(nodal_potentials)
            .iter()
            .map(|electric_field| constitutive_model.electric_displacement(electric_field))
            .collect::<Result<ElectricDisplacements<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(electric_displacements
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(electric_displacement, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector| {
                            -(electric_displacement * gradient_vector) * integration_weight
                        })
                        .collect()
                },
            )
            .sum())
    }
    fn nodal_stiffnesses(
        &self,
        constitutive_model: &C,
        nodal_potentials: &ElementNodalPotentials<N>,
    ) -> Result<ElementNodalStiffnessesElectrical<N>, FiniteElementError> {
        let electric_tangents = self
            .electric_fields(nodal_potentials)
            .iter()
            .map(|electric_field| constitutive_model.electric_tangent(electric_field))
            .collect::<Result<ElectricTangents<G>, _>>()
            .map_err(|error| FiniteElementError::upstream(error, self))?;
        Ok(electric_tangents
            .iter()
            .zip(
                self.gradient_vectors()
                    .iter()
                    .zip(self.integration_weights()),
            )
            .map(
                |(electric_tangent, (gradient_vectors, integration_weight))| {
                    gradient_vectors
                        .iter()
                        .map(|gradient_vector_a| {
                            gradient_vectors
                                .iter()
                                .map(|gradient_vector_b| {
                                    (gradient_vector_a * &(electric_tangent * gradient_vector_b))
                                        * integration_weight
                                })
                                .collect()
                        })
                        .collect()
                },
            )
            .sum())
    }
}
