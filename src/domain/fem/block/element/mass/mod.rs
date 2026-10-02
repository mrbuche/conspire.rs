#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        Element, FiniteElement,
        linear::{LinearElement, LinearFiniteElement},
    },
    math::{Quantity, Tensor, TensorList},
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

pub trait MassFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P> + ConsistentMass,
{
    fn nodal_masses(&self, densities: &IntegrationDensities<G>) -> ElementNodalMasses<P>;
}

pub trait LumpedMassFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P> + RowSumLumping,
{
    fn nodal_lumped_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> ElementNodalLumpedMasses<P>;
}

impl<const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    MassFiniteElement<G, M, N, P> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N, P> + ConsistentMass,
{
    fn nodal_masses(&self, densities: &IntegrationDensities<G>) -> ElementNodalMasses<P> {
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

impl<const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    LumpedMassFiniteElement<G, M, N, P> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N, P> + RowSumLumping,
{
    fn nodal_lumped_masses(
        &self,
        densities: &IntegrationDensities<G>,
    ) -> ElementNodalLumpedMasses<P> {
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
