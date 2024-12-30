pub mod wedge;

use super::{surface::LinearSurfaceElement, *};

pub trait LinearCohesiveElement<
    'a,
    C,
    const G: usize,
    const M: usize,
    const N: usize,
    const O: usize,
> where
    C: Cohesive<'a>,
    Self: LinearSurfaceElement<'a, C, G, M, N, O>,
{
    fn calculate_displacement(nodal_coordinates: &NodalCoordinates<N>) -> Displacement;
    fn calculate_midplane<const I: usize>(
        nodal_coordinates: &Coordinates<I, N>,
    ) -> Coordinates<I, O>;
}
