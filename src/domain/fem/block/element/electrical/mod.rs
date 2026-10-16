pub mod dielectric;

use crate::{
    fem::block::element::{Element, FiniteElement},
    math::{Quantity, Tensor, TensorList},
    mechanics::{ElectricField, ElectricFields},
    units::ElectricPotential,
};

pub type ElementNodalPotentials<const D: usize> = TensorList<Quantity<ElectricPotential>, D>;

pub trait ElectricalFiniteElement<const G: usize, const M: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N, P>,
{
    fn electric_fields(&self, nodal_potentials: &ElementNodalPotentials<N>) -> ElectricFields<G>;
}

impl<const G: usize, const M: usize, const N: usize, const O: usize, const P: usize>
    ElectricalFiniteElement<G, M, N, P> for Element<3, G, N, O>
where
    Self: FiniteElement<G, M, N, P>,
{
    fn electric_fields(&self, nodal_potentials: &ElementNodalPotentials<N>) -> ElectricFields<G> {
        self.gradient_vectors()
            .iter()
            .map(|gradient_vectors| {
                -nodal_potentials
                    .iter()
                    .zip(gradient_vectors)
                    .map(|(nodal_potential, gradient_vector)| gradient_vector * nodal_potential)
                    .sum::<ElectricField>()
            })
            .collect()
    }
}
