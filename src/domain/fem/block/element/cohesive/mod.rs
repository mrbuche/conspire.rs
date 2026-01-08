pub mod elastic;
pub mod linear;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalReferenceCoordinates, FiniteElement,
        ShapeFunctionsAtIntegrationPoints, surface::SurfaceFiniteElement,
    },
    math::{ScalarList, Tensor},
    mechanics::CurrentCoordinate,
};
use std::fmt::{self, Debug, Formatter};

pub type MidSurface<const P: usize> = ElementNodalCoordinates<P>;
pub type Separation = CurrentCoordinate;
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
    // No! Use signed shape functions, will need in nodal force and nodal stiffnesses!
    // And do not need to store!
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

impl<const G: usize, const N: usize, const O: usize, const P: usize> SurfaceFiniteElement<G, N, P>
    for CohesiveElement<G, N, O, P>
where
    Self: FiniteElement<G, M, N, P>,
{
}

pub trait CohesiveFiniteElement<const G: usize, const N: usize, const P: usize>
where
    Self: SurfaceFiniteElement<G, N, P>,
{
    fn nodal_mid_surface(nodal_coordinates: &ElementNodalCoordinates<N>) -> MidSurface<P>;
    fn nodal_separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P>;
    fn separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<G> {
        //
        // Will not work until is ShapeFunctions<P> instead of ShapeFunctions<N>.
        //
        let mid_surface = Self::nodal_mid_surface(nodal_coordinates);
        Self::shape_functions_at_integration_points()
            .iter()
            .zip(Self::full_bases(&mid_surface))
            .map(|(shape_functions, basis)| {
                let separation = Self::nodal_separations(nodal_coordinates)
                    .iter()
                    .zip(shape_functions.iter())
                    .map(|(nodal_separation, shape_function)| nodal_separation * shape_function)
                    .sum::<Separation>();
                basis
                    .into_iter()
                    .map(|basis_vector| basis_vector * &separation)
                    .collect()
            })
            .collect()
    }
    fn signed_shape_functions() -> ShapeFunctionsAtIntegrationPoints<G, N> {
        Self::shape_functions_at_integration_points()
            .iter()
            .map(|shape_functions| {
                shape_functions
                    .iter()
                    .zip(Self::signs())
                    .map(|(shape_function, sign)| shape_function * sign)
                    .collect()
            })
            .collect()
    }
    fn signs() -> ScalarList<N>;
}
