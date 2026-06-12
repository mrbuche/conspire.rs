#[cfg(test)]
pub mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalEitherCoordinates, ElementNodalReferenceCoordinates,
        FiniteElement, ParametricCoordinate, ParametricCoordinates, ParametricReference,
        ShapeFunctions, ShapeFunctionsGradients,
        cohesive::{
            CohesiveFiniteElement, M, Separations,
            linear::{LinearCohesiveElement, LinearCohesiveFiniteElement},
        },
        surface::linear::Quadrilateral,
    },
    math::ScalarList,
    mechanics::NormalGradients,
};

const G: usize = 4;
const N: usize = 8;
const P: usize = 4;

pub type Hexahedron = LinearCohesiveElement<G, N>;

impl FiniteElement<G, M, N, P> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        Quadrilateral::integration_points()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        Quadrilateral::parametric_reference()
            .into_iter()
            .chain(Quadrilateral::parametric_reference())
            .collect()
    }
    fn parametric_weights() -> ScalarList<G> {
        Quadrilateral::parametric_weights()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<P> {
        Quadrilateral::shape_functions(parametric_coordinate)
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, P> {
        Quadrilateral::shape_functions_gradients(parametric_coordinate)
    }
}

impl From<ElementNodalReferenceCoordinates<N>> for Hexahedron {
    fn from(reference_nodal_coordinates: ElementNodalReferenceCoordinates<N>) -> Self {
        Self::from_linear(reference_nodal_coordinates)
    }
}

impl CohesiveFiniteElement<G, N, P> for Hexahedron {
    fn nodal_mid_surface<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> ElementNodalEitherCoordinates<I, P> {
        Self::nodal_mid_surface_linear(nodal_coordinates)
    }
    fn nodal_separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P> {
        Self::nodal_separations_linear(nodal_coordinates)
    }
    fn normal_gradients_full(
        nodal_mid_surface: &ElementNodalCoordinates<P>,
    ) -> NormalGradients<N, G> {
        Self::normal_gradients_full_linear(nodal_mid_surface)
    }
    fn signs() -> ScalarList<N> {
        Self::signs_linear()
    }
}

impl LinearCohesiveFiniteElement<G, N, P> for Hexahedron {}
