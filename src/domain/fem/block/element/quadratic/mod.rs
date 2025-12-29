#[cfg(test)]
pub mod test;

mod pyramid;
mod tetrahedron;
mod wedge;

pub use pyramid::Pyramid;
pub use tetrahedron::Tetrahedron;
pub use wedge::Wedge;

use crate::fem::block::element::{
    Element, ElementNodalReferenceCoordinates, FiniteElement, basic_from,
};

const M: usize = 3;

pub type QuadraticElement<const G: usize, const N: usize> = Element<G, N, 2>;

pub trait QuadraticFiniteElement<const G: usize, const N: usize>
where
    Self: FiniteElement<G, M, N>,
{
}

impl<const G: usize, const N: usize> From<ElementNodalReferenceCoordinates<N>>
    for QuadraticElement<G, N>
where
    Self: QuadraticFiniteElement<G, N>,
{
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        basic_from(reference_nodal_coordinates)
    }
}
