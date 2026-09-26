#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
    fem::block::element::{Element, FiniteElement},
    math::{Current, TensorRank1List, TensorRank2List2D},
    units::{Force, ForcePerLength, Length},
};
use std::autodiff::autodiff_reverse;

type Coordinates<const D: usize, const N: usize> = TensorRank1List<D, Current, N, Length>;
type Forces<const D: usize, const N: usize> = TensorRank1List<D, Current, N, Force>;
type Stiffnesses<const D: usize, const N: usize> =
    TensorRank2List2D<D, Current, Current, N, N, ForcePerLength>;

fn component<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    x: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    let mut sum = 0.0;
    for a in 0..N {
        sum += x[D * a + i] * grad_n[base + D * a + j];
    }
    sum
}

fn entry<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    x: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    if i < D && j < D {
        component::<D, N, DOF, GN>(grad_n, base, x, i, j)
    } else if i == j {
        1.0
    } else {
        0.0
    }
}

fn deformation_gradient<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    g: usize,
    x: &[f64; DOF],
) -> [f64; 9] {
    let b = D * N * g;
    [
        entry::<D, N, DOF, GN>(grad_n, b, x, 0, 0),
        entry::<D, N, DOF, GN>(grad_n, b, x, 0, 1),
        entry::<D, N, DOF, GN>(grad_n, b, x, 0, 2),
        entry::<D, N, DOF, GN>(grad_n, b, x, 1, 0),
        entry::<D, N, DOF, GN>(grad_n, b, x, 1, 1),
        entry::<D, N, DOF, GN>(grad_n, b, x, 1, 2),
        entry::<D, N, DOF, GN>(grad_n, b, x, 2, 0),
        entry::<D, N, DOF, GN>(grad_n, b, x, 2, 1),
        entry::<D, N, DOF, GN>(grad_n, b, x, 2, 2),
    ]
}

#[autodiff_reverse(d_element_energy, Const, Const, Const, Duplicated, Active)]
fn element_energy<
    M: AutodiffHyperelastic,
    const D: usize,
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
        let f = deformation_gradient::<D, N, DOF, GN>(grad_n, g, x);
        potential += weights[g] * M::energy(parameters, &f);
    }
    potential
}

fn forces_flat<
    M: AutodiffHyperelastic,
    const D: usize,
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
    d_element_energy::<M, D, N, G, DOF, GN>(parameters, grad_n, weights, x, &mut out, 1.0);
    out
}

fn flatten<
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

fn forces<
    M: AutodiffHyperelastic,
    const D: usize,
    const G: usize,
    const N: usize,
    const O: usize,
    const DOF: usize,
    const GN: usize,
>(
    model: &M,
    element: &Element<D, G, N, O>,
    coordinates: &Coordinates<D, N>,
) -> Forces<D, N>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let flat = forces_flat::<M, D, N, G, DOF, GN>(&model.parameters(), &grad_n, &weights, &x);
    let mut out = [[0.0; D]; N];
    for a in 0..N {
        for i in 0..D {
            out[a][i] = flat[D * a + i];
        }
    }
    out.into()
}

fn stiffnesses<
    M: AutodiffHyperelastic,
    const D: usize,
    const G: usize,
    const N: usize,
    const O: usize,
    const DOF: usize,
    const GN: usize,
>(
    model: &M,
    element: &Element<D, G, N, O>,
    coordinates: &Coordinates<D, N>,
) -> Stiffnesses<D, N>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    const EPSILON: f64 = 1e-6;
    let parameters = model.parameters();
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let mut out = [[[[0.0; D]; D]; N]; N];
    for b in 0..N {
        for j in 0..D {
            let (mut plus, mut minus) = (x, x);
            plus[D * b + j] += EPSILON;
            minus[D * b + j] -= EPSILON;
            let fp = forces_flat::<M, D, N, G, DOF, GN>(&parameters, &grad_n, &weights, &plus);
            let fm = forces_flat::<M, D, N, G, DOF, GN>(&parameters, &grad_n, &weights, &minus);
            for a in 0..N {
                for i in 0..D {
                    out[a][b][i][j] = (fp[D * a + i] - fm[D * a + i]) / (2.0 * EPSILON);
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
    ($d:literal, $g:literal, $n:literal, $o:literal) => {
        impl<M> AutodiffElement<M> for Element<$d, $g, $n, $o>
        where
            M: AutodiffHyperelastic,
        {
            type Coordinates = Coordinates<$d, $n>;
            type Forces = Forces<$d, $n>;
            type Stiffnesses = Stiffnesses<$d, $n>;
            fn autodiff_nodal_forces(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Self::Forces {
                forces::<M, $d, $g, $n, $o, { $d * $n }, { $d * $n * $g }>(model, self, coordinates)
            }
            fn autodiff_nodal_stiffnesses(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Self::Stiffnesses {
                stiffnesses::<M, $d, $g, $n, $o, { $d * $n }, { $d * $n * $g }>(
                    model,
                    self,
                    coordinates,
                )
            }
        }
    };
}

shape!(3, 8, 8, 1);
shape!(3, 1, 4, 1);
shape!(2, 1, 3, 1);
