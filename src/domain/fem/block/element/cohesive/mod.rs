pub mod linear;

use crate::{
    fem::block::element::{FiniteElement, ElementNodalReferenceCoordinates},
    math::ScalarList
};
use std::fmt::{self, Debug, Formatter};

const M: usize = 2;

pub struct CohesiveElement<const G: usize, const N: usize, const O: usize, const P: usize> {
    integration_weights: ScalarList<G>,
    //
    // Store shape functions/gradients at integration points?
    // Otherwise will re-compute them each time nodal forces are evaluated.
    //
}

// no gradient vectors, and shape functions (and their gradients) are known for arbitrary elements
// do others really integrate in the current configuration?

// no deformation gradients (and not a solid), so will not need to store the reference normal either

// do not do piecewise-linear wedge-12 one, just have the linear wedge-6s placed for that

impl<const G: usize, const N: usize, const O: usize, const P: usize> Debug for CohesiveElement<G, N, O, P> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N, O, P) {
            (3, 6, 1, 3) => "LinearCohesiveWedge",
            (4, 8, 1, 4) => "LinearCohesiveHexahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize, const O: usize, const P: usize> Default for CohesiveElement<G, N, O, P>
where
    Self: FiniteElement<G, M, N> + From<ElementNodalReferenceCoordinates<P>>,
{
    fn default() -> Self {
        // (Self::parametric_reference(), 1.0).into()
        todo!()
    }
}

pub trait SurfaceFiniteElementCreation<const G: usize, const P: usize>
where
    Self: Default + From<ElementNodalReferenceCoordinates<P>>,
{
}

impl<const G: usize, const N: usize, const O: usize, const P: usize> SurfaceFiniteElementCreation<G, P>
    for CohesiveElement<G, N, O, P>
where
    Self: Default + From<ElementNodalReferenceCoordinates<P>>,
{
}
