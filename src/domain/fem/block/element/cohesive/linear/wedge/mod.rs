// #[cfg(test)]
// pub mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, FiniteElement, ParametricCoordinate, ParametricCoordinates,
        ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        cohesive::{
            CohesiveFiniteElement, M, MidSurface, Separations, linear::LinearCohesiveElement,
        },
    },
    math::{ScalarList, Tensor},
};

// This should share some methods with LinearTriangle<G=3> when get to it.

const G: usize = 3;
const N: usize = 6;
const P: usize = 3;

pub type Wedge = LinearCohesiveElement<G, N, P>;

impl FiniteElement<G, M, N> for Wedge {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [1.0 / 6.0, 1.0 / 6.0],
            [2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 2.0 / 3.0],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0 / 6.0; G].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2] = parametric_coordinate.into();
        // [1.0 - xi_1 - xi_2, xi_1, xi_2].into()
        todo!(
            "These are correct but N=6; need to use P in trait, will fix composite tet too (that one test)"
        )
    }
    fn shape_functions_gradients(
        _parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        // [[-1.0, -1.0], [1.0, 0.0], [0.0, 1.0]].into()
        todo!("These are correct but N=6...")
    }
}

impl CohesiveFiniteElement<G, N, P> for Wedge {
    fn nodal_mid_surface(nodal_coordinates: &ElementNodalCoordinates<N>) -> MidSurface<P> {
        nodal_coordinates
            .iter()
            .take(P)
            .zip(nodal_coordinates.iter().skip(P))
            .map(|(coordinates_bottom, coordinates_top)| {
                (coordinates_top + coordinates_bottom) * 0.5
            })
            .collect()
    }
    fn nodal_separations(nodal_coordinates: &ElementNodalCoordinates<N>) -> Separations<P> {
        nodal_coordinates
            .iter()
            .take(P)
            .zip(nodal_coordinates.iter().skip(P))
            .map(|(coordinates_bottom, coordinates_top)| coordinates_top - coordinates_bottom)
            .collect()
    }
    fn signs() -> ScalarList<N> {
        [1.0, 1.0, 1.0, -1.0, -1.0, -1.0].into()
    }
}
