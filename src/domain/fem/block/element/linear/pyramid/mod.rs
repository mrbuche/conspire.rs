#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalReferenceCoordinates, FiniteElement, ParametricCoordinate,
        ParametricCoordinates, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::ScalarList,
};

const G: usize = 5;
const N: usize = 5;

pub type Pyramid = LinearElement<G, N>;

impl FiniteElement<G, M, N> for Pyramid {
    fn integration_points() -> ParametricCoordinates<G, M> {
        // 5-point quadrature for pyramid
        // 4 points at mid-height in a square pattern, 1 point higher up
        // [
        //     [-0.5, -0.5, 4.0 / 30.0],
        //     [ 0.5, -0.5, 4.0 / 30.0],
        //     [ 0.5,  0.5, 4.0 / 30.0],
        //     [-0.5,  0.5, 4.0 / 30.0],
        //     [ 0.0,  0.0, 0.5],
        // ]
        // .into()
        [
        [-0.577_350_269_189_626, -0.577_350_269_189_626, 0.122_540_333_076_253],
        [ 0.577_350_269_189_626, -0.577_350_269_189_626, 0.122_540_333_076_253],
        [ 0.577_350_269_189_626,  0.577_350_269_189_626, 0.122_540_333_076_253],
        [-0.577_350_269_189_626,  0.577_350_269_189_626, 0.122_540_333_076_253],
        [ 0.0,                     0.0,                   0.544_151_844_011_225],
    ]
    .into()
    }
    
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    
    fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
        [
            [-1.0, -1.0, 0.0],
            [ 1.0, -1.0, 0.0],
            [ 1.0,  1.0, 0.0],
            [-1.0,  1.0, 0.0],
            [ 0.0,  0.0, 1.0],
        ]
        .into()
    }
    
    fn parametric_weights() -> ScalarList<G> {
        // [81.0 / 100.0 / 4.0, 81.0 / 100.0 / 4.0, 81.0 / 100.0 / 4.0, 81.0 / 100.0 / 4.0, 125.0 / 27.0 / 100.0].into()
        [
        0.148_148_148_148_148,
        0.148_148_148_148_148,
        0.148_148_148_148_148,
        0.148_148_148_148_148,
        0.407_407_407_407_407,
    ]
    .into()
    }
    
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [r, s, t] = parametric_coordinate.into();
        
        // Handle singularity at apex
        let denom = if t < 1.0 - 1e-10 { 1.0 - t } else { 1e-10 };
        
        [
            (1.0 - r - t) * (1.0 - s - t) / 4.0 / denom,
            (1.0 + r - t) * (1.0 - s - t) / 4.0 / denom,
            (1.0 + r - t) * (1.0 + s - t) / 4.0 / denom,
            (1.0 - r - t) * (1.0 + s - t) / 4.0 / denom,
            t,
        ]
        .into()
    }
    
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [r, s, t] = parametric_coordinate.into();
        
        let denom = if t < 1.0 - 1e-10 { 1.0 - t } else { 1e-10 };
        let inv_denom = 1.0 / denom;
        let inv_denom_sq = inv_denom * inv_denom;
        
        // Node 0: N0 = (1 - r - t)(1 - s - t) / 4(1 - t)
        let n0_num = (1.0 - r - t) * (1.0 - s - t) / 4.0;
        let dn0_dr = -(1.0 - s - t) / 4.0 * inv_denom;
        let dn0_ds = -(1.0 - r - t) / 4.0 * inv_denom;
        // Quotient rule: d/dt[(1-r-t)(1-s-t)/(1-t)] 
        let dn0_dt = (-(2.0 - r - s - 2.0*t) / 4.0 * (1.0 - t) + n0_num) * inv_denom_sq;
        
        // Node 1: N1 = (1 + r - t)(1 - s - t) / 4(1 - t)
        let n1_num = (1.0 + r - t) * (1.0 - s - t) / 4.0;
        let dn1_dr = (1.0 - s - t) / 4.0 * inv_denom;
        let dn1_ds = -(1.0 + r - t) / 4.0 * inv_denom;
        let dn1_dt = (-(2.0 + r - s - 2.0*t) / 4.0 * (1.0 - t) + n1_num) * inv_denom_sq;
        
        // Node 2: N2 = (1 + r - t)(1 + s - t) / 4(1 - t)
        let n2_num = (1.0 + r - t) * (1.0 + s - t) / 4.0;
        let dn2_dr = (1.0 + s - t) / 4.0 * inv_denom;
        let dn2_ds = (1.0 + r - t) / 4.0 * inv_denom;
        let dn2_dt = (-(2.0 + r + s - 2.0*t) / 4.0 * (1.0 - t) + n2_num) * inv_denom_sq;
        
        // Node 3: N3 = (1 - r - t)(1 + s - t) / 4(1 - t)
        let n3_num = (1.0 - r - t) * (1.0 + s - t) / 4.0;
        let dn3_dr = -(1.0 + s - t) / 4.0 * inv_denom;
        let dn3_ds = (1.0 - r - t) / 4.0 * inv_denom;
        let dn3_dt = (-(2.0 - r + s - 2.0*t) / 4.0 * (1.0 - t) + n3_num) * inv_denom_sq;
        
        // Node 4: N4 = t
        let dn4_dr = 0.0;
        let dn4_ds = 0.0;
        let dn4_dt = 1.0;
        
        [
            [dn0_dr, dn0_ds, dn0_dt],
            [dn1_dr, dn1_ds, dn1_dt],
            [dn2_dr, dn2_ds, dn2_dt],
            [dn3_dr, dn3_ds, dn3_dt],
            [dn4_dr, dn4_ds, dn4_dt],
        ]
        .into()
    }
}

impl LinearFiniteElement<G, N> for Pyramid {}

// impl FiniteElement<G, M, N> for Pyramid {
//     fn integration_points() -> ParametricCoordinates<G, M> {
//         [
//             [-0.5, 0.0, 1.0 / 6.0],
//             [0.5, 0.0, 1.0 / 6.0],
//             [0.0, -0.5, 1.0 / 6.0],
//             [0.0, 0.5, 1.0 / 6.0],
//             [0.0, 0.0, 0.25],
//         ]
//         .into()
//     }
//     fn integration_weights(&self) -> &ScalarList<G> {
//         &self.integration_weights
//     }
//     fn parametric_reference() -> ElementNodalReferenceCoordinates<N> {
//         [
//             [-1.0, -1.0, 0.0],
//             [1.0, -1.0, 0.0],
//             [1.0, 1.0, 0.0],
//             [-1.0, 1.0, 0.0],
//             [0.0, 0.0, 1.0],
//         ]
//         .into()
//     }
//     fn parametric_weights() -> ScalarList<G> {
//         [5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 5.0 / 27.0, 16.0 / 27.0].into()
//     }
//     fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
//         let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
//         [
//             (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
//             (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
//             (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
//             (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
//             (1.0 + xi_3) / 2.0,
//         ]
//         .into()
//     }
//     fn shape_functions_gradients(
//         parametric_coordinate: ParametricCoordinate<M>,
//     ) -> ShapeFunctionsGradients<M, N> {
//         let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
//         [
//             [
//                 -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
//                 -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
//                 -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
//             ],
//             [
//                 (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
//                 -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
//                 -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
//             ],
//             [
//                 (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
//                 (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
//                 -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
//             ],
//             [
//                 -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
//                 (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
//                 -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
//             ],
//             [0.0, 0.0, 0.5],
//         ]
//         .into()
//     }
// }

// impl LinearFiniteElement<G, N> for Pyramid {}
