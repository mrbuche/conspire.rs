#![allow(clippy::needless_range_loop)]

use crate::{
    fem::block::element::{Element, FiniteElement},
    math::{Current, TensorRank1List},
    units::Length,
};

pub(crate) type Coordinates<const D: usize, const N: usize> =
    TensorRank1List<D, Current, N, Length>;

pub(crate) fn flatten<
    const D: usize,
    const G: usize,
    const N: usize,
    const O: usize,
    const DOF: usize,
    const GN: usize,
>(
    element: &Element<D, G, N, O>,
    coordinates: &Coordinates<D, N>,
) -> ([f64; GN], [f64; G], [f64; DOF])
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    let mut grad_n = [0.0; GN];
    let mut weights = [0.0; G];
    let mut x = [0.0; DOF];
    for (g, node_gradients) in element.gradient_vectors().into_iter().enumerate() {
        for (a, gradient) in node_gradients.into_iter().enumerate() {
            for k in 0..D {
                grad_n[D * N * g + D * a + k] = gradient[k].value();
            }
        }
    }
    for (g, weight) in element.integration_weights().into_iter().enumerate() {
        weights[g] = weight.value();
    }
    for (a, coordinate) in coordinates.into_iter().enumerate() {
        for i in 0..D {
            x[D * a + i] = coordinate[i].value();
        }
    }
    (grad_n, weights, x)
}

pub(crate) fn unflatten<const D: usize, const N: usize, const DOF: usize>(
    flat: &[f64; DOF],
) -> [[f64; D]; N] {
    let mut out = [[0.0; D]; N];
    for a in 0..N {
        for i in 0..D {
            out[a][i] = flat[D * a + i];
        }
    }
    out
}

pub(crate) fn central_difference<const D: usize, const N: usize, const DOF: usize>(
    x: &[f64; DOF],
    mut flat_function: impl FnMut(&[f64; DOF]) -> [f64; DOF],
) -> [[[[f64; D]; D]; N]; N] {
    const EPSILON: f64 = 1e-6;
    let mut out = [[[[0.0; D]; D]; N]; N];
    for b in 0..N {
        for j in 0..D {
            let (mut plus, mut minus) = (*x, *x);
            plus[D * b + j] += EPSILON;
            minus[D * b + j] -= EPSILON;
            let fp = flat_function(&plus);
            let fm = flat_function(&minus);
            for a in 0..N {
                for i in 0..D {
                    out[a][b][i][j] = (fp[D * a + i] - fm[D * a + i]) / (2.0 * EPSILON);
                }
            }
        }
    }
    out
}
