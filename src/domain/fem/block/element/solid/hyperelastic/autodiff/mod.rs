//! Prototype: autodiff element residual / tangent for the linear hexahedron.
//!
//! `--features fem,autodiff` + the Enzyme toolchain (see
//! [`constitutive::solid::elastic::autodiff`](crate::constitutive::solid::elastic::autodiff)).
//!
//! The element potential `Pi(x) = sum_g w_g * Psi(F_g(x))` is a scalar `f64`
//! kernel generic over any [`AutodiffHyperelastic`] model. Reverse mode over it
//! is the nodal internal force `R_a = dPi/dx_a` (equals
//! `ElasticFiniteElement::nodal_forces`). Hard-coded to `N = G = 8` for now.
//!
//! The nodal stiffness *should* be forward-over-reverse of the same kernel, but
//! that path hits an Enzyme bug: forward mode over a reverse-mode function that
//! builds a stack `[f64; N]` from the active input and passes it by reference
//! to a callee (here `F_g` -> `M::energy`) crashes its type analysis (`Illegal
//! updateAnalysis` at `TypeTree.h:330` — the shadow of that stack array is
//! mis-sized). Reverse mode alone is fine; passing the components as scalars
//! instead of an array ref is fine. Until it is fixed, [`nodal_stiffnesses`]
//! central-differences the reverse-mode forces (`2 * DOF` reverse passes).

#![allow(clippy::needless_range_loop)]

#[cfg(test)]
mod test;

use crate::{
    constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
    fem::block::element::{Element, ElementNodalCoordinates, FiniteElement},
};
use std::autodiff::autodiff_reverse;

const N: usize = 8;
const G: usize = 8;
const DOF: usize = 3 * N;
const GN: usize = 3 * N * G;

/// `F_g[i][J] = sum_a x[3a + i] * grad_n[3N*g + 3a + J]`. Plain loop, no iterator
/// adaptors (they crash Enzyme's type analysis).
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

/// `Pi(x) = sum_g w_g * Psi(F_g(x))`, `x` the flattened nodal coordinates.
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

/// `R = dPi/dx`, reverse mode, from the flattened inputs.
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

/// Nodal internal forces (row-major, `[3a + i]`), reverse-mode AD of the element
/// potential. Matches `ElasticFiniteElement::nodal_forces`.
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

/// Nodal stiffness `[3a + i][3b + j]`: central difference of the reverse-mode
/// [`nodal_forces`] (`2 * DOF` reverse passes) while forward-over-reverse is
/// blocked by the Enzyme bug noted in the module docs.
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
        let forces_plus = forces_flat::<M>(&parameters, &grad_n, &weights, &plus);
        let forces_minus = forces_flat::<M>(&parameters, &grad_n, &weights, &minus);
        for row in 0..DOF {
            stiffness[row][column] = (forces_plus[row] - forces_minus[row]) / (2.0 * EPSILON);
        }
    }
    stiffness
}
