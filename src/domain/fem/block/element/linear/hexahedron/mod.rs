#[cfg(test)]
mod test;

use crate::{
    fem::block::element::{
        ElementNodalCoordinates, ElementNodalEitherCoordinates, FRAC_1_SQRT_3, FiniteElement,
        FiniteElementImprovement, FiniteElementMetrics, ParametricCoordinate,
        ParametricCoordinates, ParametricReference, ShapeFunctions, ShapeFunctionsGradients,
        linear::{LinearElement, LinearFiniteElement, M},
    },
    math::{Scalar, ScalarList, Tensor, TensorArray},
    mechanics::{Coordinate, ForceList, Stiffness, StiffnessList2D},
};

const G: usize = 8;
const N: usize = 8;
const P: usize = N;

const CORNERS: [[usize; 3]; N] = [
    [1, 3, 4],
    [2, 0, 5],
    [3, 1, 6],
    [0, 2, 7],
    [7, 5, 0],
    [4, 6, 1],
    [5, 7, 2],
    [6, 4, 3],
];

pub type Hexahedron = LinearElement<G, N>;

impl FiniteElement<G, M, N, P> for Hexahedron {
    fn integration_points() -> ParametricCoordinates<G, M> {
        [
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [-FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, -FRAC_1_SQRT_3, FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3, -FRAC_1_SQRT_3],
            [FRAC_1_SQRT_3, FRAC_1_SQRT_3, FRAC_1_SQRT_3],
        ]
        .into()
    }
    fn integration_weights(&self) -> &ScalarList<G> {
        &self.integration_weights
    }
    fn parametric_reference() -> ParametricReference<M, N> {
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ]
        .into()
    }
    fn parametric_weights() -> ScalarList<G> {
        [1.0; G].into()
    }
    fn shape_functions(parametric_coordinate: ParametricCoordinate<M>) -> ShapeFunctions<N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 + xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
            (1.0 - xi_1) * (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
        ]
        .into()
    }
    fn shape_functions_gradients(
        parametric_coordinate: ParametricCoordinate<M>,
    ) -> ShapeFunctionsGradients<M, N> {
        let [xi_1, xi_2, xi_3] = parametric_coordinate.into();
        [
            [
                -(1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 - xi_2) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 + xi_2) * (1.0 - xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 - xi_2) * (1.0 + xi_3) / 8.0,
                -(1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 - xi_2) / 8.0,
            ],
            [
                (1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 + xi_1) * (1.0 + xi_2) / 8.0,
            ],
            [
                -(1.0 + xi_2) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_3) / 8.0,
                (1.0 - xi_1) * (1.0 + xi_2) / 8.0,
            ],
        ]
        .into()
    }
}

impl LinearFiniteElement<G, N> for Hexahedron {}

impl FiniteElementMetrics<G, M, N, P> for Hexahedron {
    fn minimum_jacobian<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> Scalar {
        Self::jacobians(nodal_coordinates)
            .into_iter()
            .reduce(Scalar::min)
            .unwrap()
    }
    fn minimum_scaled_jacobian<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> Scalar {
        Self::scaled_jacobians(nodal_coordinates)
            .into_iter()
            .reduce(Scalar::min)
            .unwrap()
    }
}

impl FiniteElementImprovement<G, M, N, P> for Hexahedron {
    fn jacobians<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> ScalarList<N> {
        CORNERS
            .into_iter()
            .enumerate()
            .map(|(node, [node_a, node_b, node_c])| {
                let u = &nodal_coordinates[node_a] - &nodal_coordinates[node];
                let v = &nodal_coordinates[node_b] - &nodal_coordinates[node];
                let w = &nodal_coordinates[node_c] - &nodal_coordinates[node];
                u.cross(&v) * &w
            })
            .collect()
    }
    fn jacobian_gradients(
        exponent: Scalar,
        nodal_coordinates: ElementNodalCoordinates<N>,
    ) -> ForceList<N> {
        let mut weights = Self::jacobians_relative(&nodal_coordinates)
            .0
            .into_iter()
            .map(|jacobian| (-exponent * jacobian).exp())
            .collect::<ScalarList<N>>();
        weights /= weights.iter().sum::<Scalar>();
        let mut gradients = ForceList::<N>::zero();
        CORNERS.into_iter().enumerate().zip(weights).for_each(
            |((node, [node_a, node_b, node_c]), weight)| {
                let u = &nodal_coordinates[node_a] - &nodal_coordinates[node];
                let v = &nodal_coordinates[node_b] - &nodal_coordinates[node];
                let w = &nodal_coordinates[node_c] - &nodal_coordinates[node];
                let dxa = v.cross(&w);
                let dxb = w.cross(&u);
                let dxc = u.cross(&v);
                let dxi = &dxa + &dxb + &dxc;
                gradients[node_a] -= dxa * weight;
                gradients[node_b] -= dxb * weight;
                gradients[node_c] -= dxc * weight;
                gradients[node] += dxi * weight;
            },
        );
        gradients
    }
    fn jacobian_tangents(
        exponent: Scalar,
        nodal_coordinates: ElementNodalCoordinates<N>,
    ) -> StiffnessList2D<N> {
        // Force convention: F = -∇f (matches your jacobian_gradients signs)
        // Tangent (stiffness): K = ∂F/∂x = -∇² f

        // Helper: rank-2 tensor [p]_x such that [p]_x * q = p.cross(q)
        fn cross_mat<const I: usize>(p: &Coordinate<I>) -> Stiffness {
            let px = p[0];
            let py = p[1];
            let pz = p[2];
            [[0.0, -pz, py], [pz, 0.0, -px], [-py, px, 0.0]].into()
        }

        let mut weights = Self::jacobians_relative(&nodal_coordinates)
            .0
            .into_iter()
            .map(|j| (-exponent * j).exp())
            .collect::<ScalarList<N>>();
        weights /= weights.iter().sum::<Scalar>();

        // We need:
        //   G = Σ w_i g_i   where g_i = ∇J_i (24-vector, but we store as per-node 3-vectors)
        //   A = Σ w_i g_i g_i^T (block outer products)
        //   Hsum = Σ w_i H_i where H_i = ∇²J_i (block 8x8 of 3x3)
        //
        // Then K = -Hsum + β (A - G G^T)

        let mut g = ForceList::<N>::zero();
        let mut a = StiffnessList2D::<N>::zero();
        let mut hsum = StiffnessList2D::<N>::zero();

        CORNERS.into_iter().enumerate().zip(weights).for_each(
            |((i, [a_node, b_node, c_node]), wi)| {
                // edge vectors for this corner i
                let u = &nodal_coordinates[a_node] - &nodal_coordinates[i];
                let v = &nodal_coordinates[b_node] - &nodal_coordinates[i];
                let w = &nodal_coordinates[c_node] - &nodal_coordinates[i];

                // per-corner gradient of J (node-wise 3-vectors)
                // (mathematical) ∇J:
                let g_a = v.cross(&w);
                let g_b = w.cross(&u);
                let g_c = u.cross(&v);
                let g_i = -(&g_a + &g_b + &g_c);

                // accumulate g = Σ wi * ∇J_i   (still "math gradient", not force)
                g[a_node] += &g_a * wi;
                g[b_node] += &g_b * wi;
                g[c_node] += &g_c * wi;
                g[i] += &g_i * wi;

                // accumulate A = Σ wi * (∇J_i)(∇J_i)^T as blocks
                let nodes = [i, a_node, b_node, c_node];
                let grads = [g_i, g_a, g_b, g_c];
                nodes.iter().zip(grads.iter()).for_each(|(&p, grads_pp)| {
                    nodes.iter().zip(grads.iter()).for_each(|(&q, grads_qq)| {
                        a[p][q] += Stiffness::from((grads_pp, grads_qq)) * wi
                    })
                });

                // accumulate Hsum = Σ wi * ∇²J_i using 3x3 blocks
                let ux = cross_mat(&u);
                let vx = cross_mat(&v);
                let wx = cross_mat(&w);

                // Neighbor-neighbor
                // H_ab = -[w]_x ; H_ba = [w]_x
                hsum[a_node][b_node] -= &wx * wi;
                hsum[b_node][a_node] += &wx * wi;

                // H_bc = -[u]_x ; H_cb = [u]_x
                hsum[b_node][c_node] -= &ux * wi;
                hsum[c_node][b_node] += &ux * wi;

                // H_ca = -[v]_x ; H_ac = [v]_x
                hsum[c_node][a_node] -= &vx * wi;
                hsum[a_node][c_node] += &vx * wi;

                // Corner-neighbor
                // H_ia = [v]_x+[w]_x ; H_ai = -(...) etc.
                let hia = &vx - &wx;
                let hib = wx - &ux;
                let hic = ux - vx;

                hsum[i][a_node] += &hia * wi;
                hsum[a_node][i] -= hia * wi;

                hsum[i][b_node] += &hib * wi;
                hsum[b_node][i] -= hib * wi;

                hsum[i][c_node] += &hic * wi;
                hsum[c_node][i] -= hic * wi;

                // diagonal blocks are zero; no need to add
            },
        );

        // Now build K = -Hsum + β (A - G G^T), and then negate because Force = -∇f?
        // Careful with conventions:
        //
        // f = softmin(J)
        // ∇f = Σ w_i ∇J_i  = g
        // ∇²f = Hsum - β (A - g g^T)
        // Force F = -∇f
        // Tangent K = ∂F/∂x = -∇²f = -Hsum + β (A - g g^T)

        // Build GG^T in block form
        let mut gg = StiffnessList2D::<N>::zero();
        gg.iter_mut().zip(g.iter()).for_each(|(gg_p, g_p)| {
            gg_p.iter_mut()
                .zip(g.iter())
                .for_each(|(gg_pq, g_q)| *gg_pq += Stiffness::from((g_p, g_q)))
        });

        // K = -Hsum + β (A - GG)
        (a - gg) * exponent - hsum
    }
    fn scaled_jacobians<const I: usize>(
        nodal_coordinates: &ElementNodalEitherCoordinates<I, N>,
    ) -> ScalarList<N> {
        let mut u = Coordinate::zero();
        let mut v = Coordinate::zero();
        let mut w = Coordinate::zero();
        CORNERS
            .into_iter()
            .enumerate()
            .map(|(node, [node_a, node_b, node_c])| {
                u = &nodal_coordinates[node_a] - &nodal_coordinates[node];
                v = &nodal_coordinates[node_b] - &nodal_coordinates[node];
                w = &nodal_coordinates[node_c] - &nodal_coordinates[node];
                (u.cross(&v) * &w) / u.norm() / v.norm() / w.norm()
            })
            .collect()
    }
    fn scaled_jacobian_gradients(
        exponent: Scalar,
        nodal_coordinates: ElementNodalCoordinates<N>,
    ) -> ForceList<N> {
        let mut weights = Self::scaled_jacobians_relative(&nodal_coordinates)
            .0
            .into_iter()
            .map(|scaled_jacobian| (-exponent * scaled_jacobian).exp())
            .collect::<ScalarList<N>>();
        weights /= weights.iter().sum::<Scalar>();
        let mut gradients = ForceList::zero();
        CORNERS.into_iter().enumerate().zip(weights).for_each(
            |((node, [node_a, node_b, node_c]), weight)| {
                let u = &nodal_coordinates[node_a] - &nodal_coordinates[node];
                let v = &nodal_coordinates[node_b] - &nodal_coordinates[node];
                let w = &nodal_coordinates[node_c] - &nodal_coordinates[node];

                let lu2 = u.norm_squared();
                let lv2 = v.norm_squared();
                let lw2 = w.norm_squared();

                let lu = lu2.sqrt();
                let lv = lv2.sqrt();
                let lw = lw2.sqrt();

                let d = lu * lv * lw;
                let s = (u.cross(&v) * &w) / d; // THIS IS SJ

                // dJ/dx (same as unscaled Jacobian)
                let dja = v.cross(&w);
                let djb = w.cross(&u);
                let djc = u.cross(&v);
                let dji = -(&dja + &djb + &djc);

                let dln_a = &u / lu2;
                let dln_b = &v / lv2;
                let dln_c = &w / lw2;
                let dln_i = -(&dln_a + &dln_b + &dln_c);

                let dsa = dja / d - &dln_a * s;
                let dsb = djb / d - &dln_b * s;
                let dsc = djc / d - &dln_c * s;
                let dsi = dji / d - &dln_i * s;

                gradients[node_a] -= dsa * weight;
                gradients[node_b] -= dsb * weight;
                gradients[node_c] -= dsc * weight;
                gradients[node] -= dsi * weight;
            },
        );
        gradients
    }
}
