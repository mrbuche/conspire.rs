#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement,
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
    },
};
use std::autodiff::autodiff_reverse;

const N_MAX: usize = 27;
const G_MAX: usize = 27;
const DOF_MAX: usize = 3 * N_MAX;
const GN_MAX: usize = 3 * N_MAX * G_MAX;

fn component(
    grad_n: &[f64; GN_MAX],
    base: usize,
    x: &[f64; DOF_MAX],
    n: usize,
    i: usize,
    j: usize,
) -> f64 {
    let mut sum = 0.0;
    for a in 0..n {
        sum += x[3 * a + i] * grad_n[base + 3 * a + j];
    }
    sum
}

fn deformation_gradient(
    grad_n: &[f64; GN_MAX],
    g: usize,
    x: &[f64; DOF_MAX],
    n: usize,
) -> [f64; 9] {
    let b = 3 * n * g;
    [
        component(grad_n, b, x, n, 0, 0),
        component(grad_n, b, x, n, 0, 1),
        component(grad_n, b, x, n, 0, 2),
        component(grad_n, b, x, n, 1, 0),
        component(grad_n, b, x, n, 1, 1),
        component(grad_n, b, x, n, 1, 2),
        component(grad_n, b, x, n, 2, 0),
        component(grad_n, b, x, n, 2, 1),
        component(grad_n, b, x, n, 2, 2),
    ]
}

#[autodiff_reverse(
    d_element_energy,
    Const,
    Const,
    Const,
    Duplicated,
    Const,
    Const,
    Active
)]
fn element_energy<M: AutodiffHyperelastic>(
    parameters: &[f64; 2],
    grad_n: &[f64; GN_MAX],
    weights: &[f64; G_MAX],
    x: &[f64; DOF_MAX],
    n: usize,
    g: usize,
) -> f64 {
    let mut potential = 0.0;
    for k in 0..g {
        let f = deformation_gradient(grad_n, k, x, n);
        potential += weights[k] * M::energy(parameters, &f);
    }
    potential
}

fn forces_flat<M: AutodiffHyperelastic>(
    parameters: &[f64; 2],
    grad_n: &[f64; GN_MAX],
    weights: &[f64; G_MAX],
    x: &[f64; DOF_MAX],
    n: usize,
    g: usize,
) -> [f64; DOF_MAX] {
    let mut out = [0.0; DOF_MAX];
    d_element_energy::<M>(parameters, grad_n, weights, x, &mut out, n, g, 1.0);
    out
}

struct Flat {
    grad_n: [f64; GN_MAX],
    weights: [f64; G_MAX],
    x: [f64; DOF_MAX],
}

fn flatten<const G: usize, const N: usize, const O: usize>(
    element: &Element<3, G, N, O>,
    coordinates: &ElementNodalCoordinates<N>,
) -> Flat
where
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    const { assert!(N <= N_MAX && G <= G_MAX) };
    let mut flat = Flat {
        grad_n: [0.0; GN_MAX],
        weights: [0.0; G_MAX],
        x: [0.0; DOF_MAX],
    };
    for (g, node_gradients) in element.gradient_vectors().into_iter().enumerate() {
        for (a, gradient) in node_gradients.into_iter().enumerate() {
            for k in 0..3 {
                flat.grad_n[3 * N * g + 3 * a + k] = gradient[k].value();
            }
        }
    }
    for (g, weight) in element.integration_weights().into_iter().enumerate() {
        flat.weights[g] = weight.value();
    }
    for (a, coordinate) in coordinates.into_iter().enumerate() {
        for i in 0..3 {
            flat.x[3 * a + i] = coordinate[i].value();
        }
    }
    flat
}

pub fn nodal_forces<M, const G: usize, const N: usize, const O: usize>(
    model: &M,
    element: &Element<3, G, N, O>,
    coordinates: &ElementNodalCoordinates<N>,
) -> ElementNodalForcesSolid<N>
where
    M: AutodiffHyperelastic,
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    let flat = flatten(element, coordinates);
    let forces = forces_flat::<M>(
        &model.parameters(),
        &flat.grad_n,
        &flat.weights,
        &flat.x,
        N,
        G,
    );
    let mut out = [[0.0; 3]; N];
    for a in 0..N {
        for i in 0..3 {
            out[a][i] = forces[3 * a + i];
        }
    }
    out.into()
}

pub fn nodal_stiffnesses<M, const G: usize, const N: usize, const O: usize>(
    model: &M,
    element: &Element<3, G, N, O>,
    coordinates: &ElementNodalCoordinates<N>,
) -> ElementNodalStiffnessesSolid<N>
where
    M: AutodiffHyperelastic,
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    const EPSILON: f64 = 1e-6;
    let parameters = model.parameters();
    let flat = flatten(element, coordinates);
    let mut out = [[[[0.0; 3]; 3]; N]; N];
    for b in 0..N {
        for j in 0..3 {
            let (mut plus, mut minus) = (flat.x, flat.x);
            plus[3 * b + j] += EPSILON;
            minus[3 * b + j] -= EPSILON;
            let fp = forces_flat::<M>(&parameters, &flat.grad_n, &flat.weights, &plus, N, G);
            let fm = forces_flat::<M>(&parameters, &flat.grad_n, &flat.weights, &minus, N, G);
            for a in 0..N {
                for i in 0..3 {
                    out[a][b][i][j] = (fp[3 * a + i] - fm[3 * a + i]) / (2.0 * EPSILON);
                }
            }
        }
    }
    out.into()
}
