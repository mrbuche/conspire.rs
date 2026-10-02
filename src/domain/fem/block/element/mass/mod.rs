#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{Element, FiniteElement},
    math::{Quantity, Tensor, TensorList},
    units::Volume,
};

pub type ElementNodalMasses<const D: usize> = TensorList<TensorList<Quantity<Volume>, D>, D>;

pub trait ConsistentMass {}

pub trait MassFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P> + ConsistentMass,
{
    fn nodal_masses(&self) -> ElementNodalMasses<P>;
}

impl<const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    MassFiniteElement<G, M, N, P> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N, P> + ConsistentMass,
{
    fn nodal_masses(&self) -> ElementNodalMasses<P> {
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(self.integration_weights())
            .map(|(shape_functions, integration_weight)| {
                (0..P)
                    .map(|a| {
                        (0..P)
                            .map(|b| integration_weight * (shape_functions[a] * shape_functions[b]))
                            .collect()
                    })
                    .collect()
            })
            .sum()
    }
}
