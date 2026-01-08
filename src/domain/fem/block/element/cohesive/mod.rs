pub mod elastic;
pub mod linear;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
    },
    math::{ScalarList, Tensor},
};
use std::fmt::{self, Debug, Formatter};

pub type MidSurface<const P: usize> = ElementNodalCoordinates<P>;
pub type Separations<const P: usize> = ElementNodalCoordinates<P>;

const M: usize = 2;

pub struct CohesiveElement<const G: usize, const N: usize, const O: usize, const P: usize> {
    integration_weights: ScalarList<G>,
    //
    // Store shape functions/gradients at integration points?
    // Otherwise will re-compute them each time nodal forces are evaluated.
    //
    // No, do not store, they are the same for every element!
    // Override the shape_functions_at_integration_points() default trait impl instead!
    //
}

// no gradient vectors, and shape functions (and their gradients) are known for arbitrary elements
// do others really integrate in the current configuration?

// no deformation gradients (and not a solid), so will not need to store the reference normal either

// do not do piecewise-linear wedge-12 one, just have the linear wedge-6s placed for that

impl<const G: usize, const N: usize, const O: usize, const P: usize> Debug
    for CohesiveElement<G, N, O, P>
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let element = match (G, N, O, P) {
            (3, 6, 1, 3) => "LinearCohesiveWedge",
            (4, 8, 1, 4) => "LinearCohesiveHexahedron",
            _ => panic!(),
        };
        write!(f, "{element} {{ G: {G}, N: {N} }}",)
    }
}

impl<const G: usize, const N: usize, const O: usize, const P: usize> Default
    for CohesiveElement<G, N, O, P>
where
    Self: FiniteElement<G, M, N> + From<ElementNodalReferenceCoordinates<P>>,
{
    fn default() -> Self {
        // (Self::parametric_reference(), 1.0).into()
        todo!()
    }
}

pub trait CohesiveFiniteElementCreation<const G: usize, const P: usize>
where
    Self: Default + From<ElementNodalReferenceCoordinates<P>>,
{
}

impl<const G: usize, const N: usize, const O: usize, const P: usize>
    CohesiveFiniteElementCreation<G, P> for CohesiveElement<G, N, O, P>
where
    Self: Default + From<ElementNodalReferenceCoordinates<P>>,
{
}

pub trait CohesiveFiniteElement<const G: usize, const N: usize, const P: usize>
where
    Self: FiniteElement<G, M, N>,
{
    fn nodal_mid_surface(nodal_coordinates: &ElementNodalCoordinates<N>) -> MidSurface<P>;
    fn nodal_separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P>;
    fn separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<G> {
        //
        // Will not work until is ShapeFunctions<P> instead of ShapeFunctions<N>.
        //
        Self::shape_functions_at_integration_points()
            .iter()
            .map(|shape_functions| {
                Self::nodal_separations(nodal_coordinates)
                    .iter()
                    .zip(shape_functions.iter())
                    .map(|(nodal_separation, shape_function)| nodal_separation * shape_function)
                    .sum()
            })
            .collect()
    }
}
