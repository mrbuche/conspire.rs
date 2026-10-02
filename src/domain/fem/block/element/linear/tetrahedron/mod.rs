#[cfg(feature = "autodiff")]
mod autodiff;
#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        FiniteElement, IntegrationWeights, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
        mass::ConsistentMass,
    },
    math::ScalarList,
    units::Volume,
};

const G1: usize = 1;
const G4: usize = 4;
const N: usize = 4;
const P: usize = N;

pub type Tetrahedron<const G: usize> = LinearElement<G, N>;

impl FiniteElement<G1, M, N, P> for Tetrahedron<G1> {
    fn integration_points() -> ParametricCoordinates<G1, M> {
        [[0.25; M]].into()
    }
    fn integration_weights(&self) -> &IntegrationWeights<G1, Volume> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G1> {
        [1.0 / 6.0; G1].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [1.0 - xi_1 - xi_2 - xi_3, xi_1, xi_2, xi_3].into()
    }
    fn shape_functions_gradients(
        _parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        [
            [-1.0, -1.0, -1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        .into()
    }
}

impl FiniteElement<G4, M, N, P> for Tetrahedron<G4> {
    fn integration_points() -> ParametricCoordinates<G4, M> {
        let alpha = (5.0 + 3.0 * 5.0_f64.sqrt()) / 20.0;
        let beta = (5.0 - 5.0_f64.sqrt()) / 20.0;
        [
            [beta, beta, beta],
            [alpha, beta, beta],
            [beta, alpha, beta],
            [beta, beta, alpha],
        ]
        .into()
    }
    fn integration_weights(&self) -> &IntegrationWeights<G4, Volume> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        Tetrahedron::<G1>::parametric_reference()
    }
    fn parametric_weights() -> ScalarList<G4> {
        [1.0 / 24.0; G4].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        Tetrahedron::<G1>::shape_functions(parametric_coordinate)
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        Tetrahedron::<G1>::shape_functions_gradients(parametric_coordinate)
    }
}

impl LinearFiniteElement<G1, N> for Tetrahedron<G1> {}

impl LinearFiniteElement<G4, N> for Tetrahedron<G4> {}

impl ConsistentMass for Tetrahedron<G4> {}
