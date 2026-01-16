mod hexahedron;
mod pyramid;
mod tetrahedron;
mod wedge;

pub use hexahedron::Hexahedron;
pub use pyramid::Pyramid;
pub use tetrahedron::Tetrahedron;
pub use wedge::Wedge;

use crate::fem::block::element::{
    Element, ElementNodalReferenceCoordinates, FiniteElement, basic_from,
};

const M: usize = 3;

pub type LinearElement<const G: usize, const N: usize> = Element<G, N, 1>;

pub trait LinearFiniteElement<const G: usize, const N: usize>
where
    Self: FiniteElement<G, M, N, N>,
{
}

impl<const G: usize, const N: usize> From<ElementNodalReferenceCoordinates<N>>
    for LinearElement<G, N>
where
    Self: LinearFiniteElement<G, N>,
{
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        basic_from(reference_nodal_coordinates)
    }
}
