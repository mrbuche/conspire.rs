#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FRAC_1_SQRT_3, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        quadratic::{QuadraticElement, QuadraticFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 27;
const N: usize = 27;

pub type Hexahedron = QuadraticElement<G, N>;

impl FiniteElement<G, M, N> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        todo!()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        todo!()
    }
    fn parametric_weights() -> ScalarList<G> {
        todo!()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        todo!()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        todo!()
    }
}

impl QuadraticFiniteElement<G, N> for Hexahedron {}
