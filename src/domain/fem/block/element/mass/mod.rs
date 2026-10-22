#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        Element, FiniteElement, FiniteElementIntegration, FiniteElementInterpolation,
        linear::{LinearElement, LinearFiniteElement},
    },
    math::{Quantity, Scalar, Tensor, TensorList},
    units::{Density, Mass},
};

pub type IntegrationDensities<const G: usize> = TensorList<Quantity<Density>, G>;
pub type ElementNodalMasses<const D: usize> = TensorList<TensorList<Quantity<Mass>, D>, D>;
pub type ElementNodalLumpedMasses<const D: usize> = TensorList<Quantity<Mass>, D>;

pub trait ConsistentMass {}

pub trait RowSumLumping {}

impl<const G: usize, const N: usize> RowSumLumping for LinearElement<G, N> where
    Self: LinearFiniteElement<G, N>
{
}

pub trait DiagonalScaling {}

impl<const G: usize, const N: usize> DiagonalScaling for LinearElement<G, N> where
    Self: LinearFiniteElement<G, N>
{
}

pub trait MassFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N> + ConsistentMass,
{
    fn nodal_masses(&self, densities: &IntegrationDensities<G>) -> ElementNodalMasses<N>;
}

pub trait DiagonallyScaledMassFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N> + DiagonalScaling,
{
    fn nodal_diagonally_scaled_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> ElementNodalLumpedMasses<N>;
}

pub trait LumpedMassFiniteElement<const G: usize, const M: usize, const N: usize>
where
    Self: FiniteElement<G, M, N> + RowSumLumping,
{
    fn nodal_lumped_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> ElementNodalLumpedMasses<N>;
}

impl<const G: usize, const M: usize, const N: usize, const O: usize> MassFiniteElement<G, M, N>
    for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N> + ConsistentMass,
{
    fn nodal_masses(&self, densities: &IntegrationDensities<G>) -> ElementNodalMasses<N> {
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(densities)
            .zip(self.integration_weights())
            .map(|((shape_functions, density), integration_weight)| {
                let mass = density * integration_weight;
                shape_functions
                    .iter()
                    .map(|shape_function_a| {
                        shape_functions
                            .iter()
                            .map(|shape_function_b| mass * (shape_function_a * shape_function_b))
                            .collect()
                    })
                    .collect()
            })
            .sum()
    }
}

impl<const G: usize, const M: usize, const N: usize, const O: usize>
    LumpedMassFiniteElement<G, M, N> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N> + RowSumLumping,
{
    fn nodal_lumped_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> ElementNodalLumpedMasses<N> {
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(densities)
            .zip(self.integration_weights())
            .map(|((shape_functions, density), integration_weight)| {
                let mass = density * integration_weight;
                shape_functions
                    .iter()
                    .map(|shape_function| mass * shape_function)
                    .collect()
            })
            .sum()
    }
}

impl<const G: usize, const M: usize, const N: usize, const O: usize>
    DiagonallyScaledMassFiniteElement<G, M, N> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N> + DiagonalScaling,
{
    fn nodal_diagonally_scaled_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> ElementNodalLumpedMasses<N> {
        let diagonals: ElementNodalLumpedMasses<N> = Self::shape_functions_at_integration_points()
            .iter()
            .zip(densities)
            .zip(self.integration_weights())
            .map(|((shape_functions, density), integration_weight)| {
                let mass = density * integration_weight;
                shape_functions
                    .iter()
                    .map(|shape_function| mass * (shape_function * shape_function))
                    .collect()
            })
            .sum();
        let total = densities
            .iter()
            .zip(self.integration_weights())
            .map(|(density, integration_weight)| density * integration_weight)
            .sum::<Quantity<Mass>>();
        let scale = total.value()
            / diagonals
                .iter()
                .map(|diagonal| diagonal.value())
                .sum::<Scalar>();
        diagonals.iter().map(|diagonal| *diagonal * scale).collect()
    }
}
