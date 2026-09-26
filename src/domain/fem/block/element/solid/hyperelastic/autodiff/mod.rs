#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
    fem::block::element::{Element, ElementNodalCoordinates, FiniteElement},
};
use std::autodiff::autodiff_reverse;

const N: usize = 8;
const G: usize = 8;
const DOF: usize = 3 * N;
const GN: usize = 3 * N * G;

fn component(grad_n: &[f64; GN], base: usize, x: &[f64; DOF], i: usize, j: usize) -> f64 {
    let mut sum = 0.0;
    for a in 0..N {
        sum += x[3 * a + i] * grad_n[base + 3 * a + j];
    }
    sum
}

fn deformation_gradient(grad_n: &[f64; GN], g: usize, x: &[f64; DOF]) -> [f64; 9] {
    let b = 3 * N * g;
    [
        component(grad_n, b, x, 0, 0),
        component(grad_n, b, x, 0, 1),
        component(grad_n, b, x, 0, 2),
        component(grad_n, b, x, 1, 0),
        component(grad_n, b, x, 1, 1),
        component(grad_n, b, x, 1, 2),
        component(grad_n, b, x, 2, 0),
        component(grad_n, b, x, 2, 1),
        component(grad_n, b, x, 2, 2),
    ]
}

#[autodiff_reverse(d_element_energy, Const, Const, Const, Duplicated, Active)]
fn element_energy<M: AutodiffHyperelastic>(
    parameters: &[f64; 2],
    grad_n: &[f64; GN],
    weights: &[f64; G],
    x: &[f64; DOF],
) -> f64 {
    let mut potential = 0.0;
    for g in 0..G {
        let f = deformation_gradient(grad_n, g, x);
        potential += weights[g] * M::energy(parameters, &f);
    }
    potential
}

fn forces_flat<M: AutodiffHyperelastic>(
    parameters: &[f64; 2],
    grad_n: &[f64; GN],
    weights: &[f64; G],
    x: &[f64; DOF],
) -> [f64; DOF] {
    let mut out = [0.0; DOF];
    d_element_energy::<M>(parameters, grad_n, weights, x, &mut out, 1.0);
    out
}

fn gradient_vectors_flat(element: &Element<3, G, N, 1>) -> [f64; GN] {
    let mut out = [0.0; GN];
    for (g, node_gradients) in element.gradient_vectors().into_iter().enumerate() {
        for (a, gradient) in node_gradients.into_iter().enumerate() {
            for k in 0..3 {
                out[3 * N * g + 3 * a + k] = gradient[k].value();
            }
        }
    }
    out
}

fn weights_flat(element: &Element<3, G, N, 1>) -> [f64; G] {
    let mut out = [0.0; G];
    for (g, weight) in element.integration_weights().into_iter().enumerate() {
        out[g] = weight.value();
    }
    out
}

fn coordinates_flat(coordinates: &ElementNodalCoordinates<N>) -> [f64; DOF] {
    let mut out = [0.0; DOF];
    for (a, coordinate) in coordinates.into_iter().enumerate() {
        for i in 0..3 {
            out[3 * a + i] = coordinate[i].value();
        }
    }
    out
}

pub fn nodal_forces<M: AutodiffHyperelastic>(
    model: &M,
    element: &Element<3, G, N, 1>,
    coordinates: &ElementNodalCoordinates<N>,
) -> [f64; DOF] {
    forces_flat::<M>(
        &model.parameters(),
        &gradient_vectors_flat(element),
        &weights_flat(element),
        &coordinates_flat(coordinates),
    )
}

pub fn nodal_stiffnesses<M: AutodiffHyperelastic>(
    model: &M,
    element: &Element<3, G, N, 1>,
    coordinates: &ElementNodalCoordinates<N>,
) -> [[f64; DOF]; DOF] {
    const EPSILON: f64 = 1e-6;
    let parameters = model.parameters();
    let (grad_n, weights) = (gradient_vectors_flat(element), weights_flat(element));
    let x = coordinates_flat(coordinates);
    let mut stiffness = [[0.0; DOF]; DOF];
    for column in 0..DOF {
        let (mut plus, mut minus) = (x, x);
        plus[column] += EPSILON;
        minus[column] -= EPSILON;
        let fp = forces_flat::<M>(&parameters, &grad_n, &weights, &plus);
        let fm = forces_flat::<M>(&parameters, &grad_n, &weights, &minus);
        for row in 0..DOF {
            stiffness[row][column] = (fp[row] - fm[row]) / (2.0 * EPSILON);
        }
    }
    stiffness
}
