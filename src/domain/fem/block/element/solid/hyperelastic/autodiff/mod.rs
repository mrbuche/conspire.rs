#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
    fem::block::element::{
        Element, ElementNodalCoordinates, FiniteElement,
        solid::{ElementNodalForcesSolid, ElementNodalStiffnessesSolid},
    },
};
use std::autodiff::autodiff_reverse;

fn component<const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    x: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    let mut sum = 0.0;
    for a in 0..N {
        sum += x[3 * a + i] * grad_n[base + 3 * a + j];
    }
    sum
}

fn deformation_gradient<const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    g: usize,
    x: &[f64; DOF],
) -> [f64; 9] {
    let b = 3 * N * g;
    [
        component::<N, DOF, GN>(grad_n, b, x, 0, 0),
        component::<N, DOF, GN>(grad_n, b, x, 0, 1),
        component::<N, DOF, GN>(grad_n, b, x, 0, 2),
        component::<N, DOF, GN>(grad_n, b, x, 1, 0),
        component::<N, DOF, GN>(grad_n, b, x, 1, 1),
        component::<N, DOF, GN>(grad_n, b, x, 1, 2),
        component::<N, DOF, GN>(grad_n, b, x, 2, 0),
        component::<N, DOF, GN>(grad_n, b, x, 2, 1),
        component::<N, DOF, GN>(grad_n, b, x, 2, 2),
    ]
}

#[autodiff_reverse(d_element_energy, Const, Const, Const, Duplicated, Active)]
fn element_energy<
    M: AutodiffHyperelastic,
    const N: usize,
    const G: usize,
    const DOF: usize,
    const GN: usize,
>(
    parameters: &[f64; 2],
    grad_n: &[f64; GN],
    weights: &[f64; G],
    x: &[f64; DOF],
) -> f64 {
    let mut potential = 0.0;
    for g in 0..G {
        let f = deformation_gradient::<N, DOF, GN>(grad_n, g, x);
        potential += weights[g] * M::energy(parameters, &f);
    }
    potential
}

fn forces_flat<
    M: AutodiffHyperelastic,
    const N: usize,
    const G: usize,
    const DOF: usize,
    const GN: usize,
>(
    parameters: &[f64; 2],
    grad_n: &[f64; GN],
    weights: &[f64; G],
    x: &[f64; DOF],
) -> [f64; DOF] {
    let mut out = [0.0; DOF];
    d_element_energy::<M, N, G, DOF, GN>(parameters, grad_n, weights, x, &mut out, 1.0);
    out
}

fn flatten<const G: usize, const N: usize, const O: usize, const DOF: usize, const GN: usize>(
    element: &Element<3, G, N, O>,
    coordinates: &ElementNodalCoordinates<N>,
) -> ([f64; GN], [f64; G], [f64; DOF])
where
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    let mut grad_n = [0.0; GN];
    let mut weights = [0.0; G];
    let mut x = [0.0; DOF];
    for (g, node_gradients) in element.gradient_vectors().into_iter().enumerate() {
        for (a, gradient) in node_gradients.into_iter().enumerate() {
            for k in 0..3 {
                grad_n[3 * N * g + 3 * a + k] = gradient[k].value();
            }
        }
    }
    for (g, weight) in element.integration_weights().into_iter().enumerate() {
        weights[g] = weight.value();
    }
    for (a, coordinate) in coordinates.into_iter().enumerate() {
        for i in 0..3 {
            x[3 * a + i] = coordinate[i].value();
        }
    }
    (grad_n, weights, x)
}

fn forces<
    M: AutodiffHyperelastic,
    const G: usize,
    const N: usize,
    const O: usize,
    const DOF: usize,
    const GN: usize,
>(
    model: &M,
    element: &Element<3, G, N, O>,
    coordinates: &ElementNodalCoordinates<N>,
) -> ElementNodalForcesSolid<N>
where
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    let (grad_n, weights, x) = flatten::<G, N, O, DOF, GN>(element, coordinates);
    let flat = forces_flat::<M, N, G, DOF, GN>(&model.parameters(), &grad_n, &weights, &x);
    let mut out = [[0.0; 3]; N];
    for a in 0..N {
        for i in 0..3 {
            out[a][i] = flat[3 * a + i];
        }
    }
    out.into()
}

fn stiffnesses<
    M: AutodiffHyperelastic,
    const G: usize,
    const N: usize,
    const O: usize,
    const DOF: usize,
    const GN: usize,
>(
    model: &M,
    element: &Element<3, G, N, O>,
    coordinates: &ElementNodalCoordinates<N>,
) -> ElementNodalStiffnessesSolid<N>
where
    Element<3, G, N, O>: FiniteElement<G, 3, N, N>,
{
    const EPSILON: f64 = 1e-6;
    let parameters = model.parameters();
    let (grad_n, weights, x) = flatten::<G, N, O, DOF, GN>(element, coordinates);
    let mut out = [[[[0.0; 3]; 3]; N]; N];
    for b in 0..N {
        for j in 0..3 {
            let (mut plus, mut minus) = (x, x);
            plus[3 * b + j] += EPSILON;
            minus[3 * b + j] -= EPSILON;
            let fp = forces_flat::<M, N, G, DOF, GN>(&parameters, &grad_n, &weights, &plus);
            let fm = forces_flat::<M, N, G, DOF, GN>(&parameters, &grad_n, &weights, &minus);
            for a in 0..N {
                for i in 0..3 {
                    out[a][b][i][j] = (fp[3 * a + i] - fm[3 * a + i]) / (2.0 * EPSILON);
                }
            }
        }
    }
    out.into()
}

pub trait AutodiffElement<M>
where
    M: AutodiffHyperelastic,
{
    type Coordinates;
    type Forces;
    type Stiffnesses;
    fn autodiff_nodal_forces(&self, model: &M, coordinates: &Self::Coordinates) -> Self::Forces;
    fn autodiff_nodal_stiffnesses(
        &self,
        model: &M,
        coordinates: &Self::Coordinates,
    ) -> Self::Stiffnesses;
}

macro_rules! shape {
    ($g:literal, $n:literal, $o:literal) => {
        impl<M> AutodiffElement<M> for Element<3, $g, $n, $o>
        where
            M: AutodiffHyperelastic,
        {
            type Coordinates = ElementNodalCoordinates<$n>;
            type Forces = ElementNodalForcesSolid<$n>;
            type Stiffnesses = ElementNodalStiffnessesSolid<$n>;
            fn autodiff_nodal_forces(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Self::Forces {
                forces::<M, $g, $n, $o, { 3 * $n }, { 3 * $n * $g }>(model, self, coordinates)
            }
            fn autodiff_nodal_stiffnesses(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Self::Stiffnesses {
                stiffnesses::<M, $g, $n, $o, { 3 * $n }, { 3 * $n * $g }>(model, self, coordinates)
            }
        }
    };
}

shape!(8, 8, 1);
shape!(1, 4, 1);
