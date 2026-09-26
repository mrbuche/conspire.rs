#![allow(clippy::needless_range_loop)]

use super::{
    Coordinates, Forces, component, deformation_gradient, element_energy, flatten, forces_flat,
};
use crate::{
    constitutive::{
        fluid::hyperviscous::autodiff::AutodiffHyperviscous,
        solid::hyperviscoelastic::autodiff::AutodiffHyperviscoelastic,
    },
    fem::block::element::{
        Element, ElementNodalCoordinates, ElementNodalVelocities, FiniteElement,
        solid::ElementNodalDampingsSolid,
    },
    math::{Current, Quantity, TensorRank1List, TensorRank2List2D},
    units::{Energy, ForcePerVelocity, Power, Velocity},
};
use std::autodiff::autodiff_reverse;

type Velocities<const D: usize, const N: usize> = TensorRank1List<D, Current, N, Velocity>;
type Dampings<const D: usize, const N: usize> =
    TensorRank2List2D<D, Current, Current, N, N, ForcePerVelocity>;

fn rate_entry<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    base: usize,
    v: &[f64; DOF],
    i: usize,
    j: usize,
) -> f64 {
    if i < D && j < D {
        component::<D, N, DOF, GN>(grad_n, base, v, i, j)
    } else {
        0.0
    }
}

fn deformation_gradient_rate<const D: usize, const N: usize, const DOF: usize, const GN: usize>(
    grad_n: &[f64; GN],
    g: usize,
    v: &[f64; DOF],
) -> [f64; 9] {
    let b = D * N * g;
    [
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 0, 0),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 0, 1),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 0, 2),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 1, 0),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 1, 1),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 1, 2),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 2, 0),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 2, 1),
        rate_entry::<D, N, DOF, GN>(grad_n, b, v, 2, 2),
    ]
}

#[autodiff_reverse(d_element_dissipation, Const, Const, Const, Const, Duplicated, Active)]
fn element_dissipation<
    V: AutodiffHyperviscous,
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
    v: &[f64; DOF],
) -> f64 {
    let mut potential = 0.0;
    for g in 0..G {
        let f = deformation_gradient::<D, N, DOF, GN>(grad_n, g, x);
        let f_dot = deformation_gradient_rate::<D, N, DOF, GN>(grad_n, g, v);
        potential += weights[g] * V::dissipation(parameters, &f, &f_dot);
    }
    potential
}

fn viscous_forces_flat<
    V: AutodiffHyperviscous,
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
    v: &[f64; DOF],
) -> [f64; DOF] {
    let mut out = [0.0; DOF];
    d_element_dissipation::<V, D, N, G, DOF, GN>(parameters, grad_n, weights, x, v, &mut out, 1.0);
    out
}

fn flatten_velocities<const D: usize, const N: usize, const DOF: usize>(
    velocities: &Velocities<D, N>,
) -> [f64; DOF] {
    let mut v = [0.0; DOF];
    for (a, velocity) in velocities.into_iter().enumerate() {
        for i in 0..D {
            v[D * a + i] = velocity[i].value();
        }
    }
    v
}

fn forces<
    M: AutodiffHyperviscoelastic,
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
    velocities: &Velocities<D, N>,
) -> Forces<D, N>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let v = flatten_velocities::<D, N, DOF>(velocities);
    let elastic = forces_flat::<M::Elastic, D, N, G, DOF, GN>(
        &model.elastic_parameters(),
        &grad_n,
        &weights,
        &x,
    );
    let viscous = viscous_forces_flat::<M::Viscous, D, N, G, DOF, GN>(
        &model.viscous_parameters(),
        &grad_n,
        &weights,
        &x,
        &v,
    );
    let mut out = [[0.0; D]; N];
    for a in 0..N {
        for i in 0..D {
            out[a][i] = elastic[D * a + i] + viscous[D * a + i];
        }
    }
    out.into()
}

fn dampings<
    M: AutodiffHyperviscoelastic,
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
    velocities: &Velocities<D, N>,
) -> Dampings<D, N>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    const EPSILON: f64 = 1e-6;
    let parameters = model.viscous_parameters();
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let v = flatten_velocities::<D, N, DOF>(velocities);
    let mut out = [[[[0.0; D]; D]; N]; N];
    for b in 0..N {
        for j in 0..D {
            let (mut plus, mut minus) = (v, v);
            plus[D * b + j] += EPSILON;
            minus[D * b + j] -= EPSILON;
            let fp = viscous_forces_flat::<M::Viscous, D, N, G, DOF, GN>(
                &parameters,
                &grad_n,
                &weights,
                &x,
                &plus,
            );
            let fm = viscous_forces_flat::<M::Viscous, D, N, G, DOF, GN>(
                &parameters,
                &grad_n,
                &weights,
                &x,
                &minus,
            );
            for a in 0..N {
                for i in 0..D {
                    out[a][b][i][j] = (fp[D * a + i] - fm[D * a + i]) / (2.0 * EPSILON);
                }
            }
        }
    }
    out.into()
}

fn viscous_dissipation<
    M: AutodiffHyperviscoelastic,
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
    velocities: &Velocities<D, N>,
) -> Quantity<Power>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let v = flatten_velocities::<D, N, DOF>(velocities);
    Quantity::new(element_dissipation::<M::Viscous, D, N, G, DOF, GN>(
        &model.viscous_parameters(),
        &grad_n,
        &weights,
        &x,
        &v,
    ))
}

fn helmholtz_free_energy<
    M: AutodiffHyperviscoelastic,
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
) -> Quantity<Energy>
where
    Element<D, G, N, O>: FiniteElement<G, D, N, N>,
{
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    Quantity::new(element_energy::<M::Elastic, D, N, G, DOF, GN>(
        &model.elastic_parameters(),
        &grad_n,
        &weights,
        &x,
    ))
}

pub trait AutodiffViscoelasticElement<M>
where
    M: AutodiffHyperviscoelastic,
{
    type Coordinates;
    type Velocities;
    type Forces;
    type Dampings;
    fn autodiff_viscoelastic_nodal_forces(
        &self,
        model: &M,
        coordinates: &Self::Coordinates,
        velocities: &Self::Velocities,
    ) -> Self::Forces;
    fn autodiff_nodal_dampings(
        &self,
        model: &M,
        coordinates: &Self::Coordinates,
        velocities: &Self::Velocities,
    ) -> Self::Dampings;
    fn autodiff_viscous_dissipation(
        &self,
        model: &M,
        coordinates: &Self::Coordinates,
        velocities: &Self::Velocities,
    ) -> Quantity<Power>;
    fn autodiff_helmholtz_free_energy(
        &self,
        model: &M,
        coordinates: &Self::Coordinates,
    ) -> Quantity<Energy>;
}

macro_rules! shape {
    ($g:literal, $n:literal, $o:literal) => {
        impl<M> AutodiffViscoelasticElement<M> for Element<3, $g, $n, $o>
        where
            M: AutodiffHyperviscoelastic,
        {
            type Coordinates = ElementNodalCoordinates<$n>;
            type Velocities = ElementNodalVelocities<$n>;
            type Forces = Forces<3, $n>;
            type Dampings = ElementNodalDampingsSolid<$n>;
            fn autodiff_viscoelastic_nodal_forces(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
                velocities: &Self::Velocities,
            ) -> Self::Forces {
                forces::<M, 3, $g, $n, $o, { 3 * $n }, { 3 * $n * $g }>(
                    model,
                    self,
                    coordinates,
                    velocities,
                )
            }
            fn autodiff_nodal_dampings(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
                velocities: &Self::Velocities,
            ) -> Self::Dampings {
                dampings::<M, 3, $g, $n, $o, { 3 * $n }, { 3 * $n * $g }>(
                    model,
                    self,
                    coordinates,
                    velocities,
                )
            }
            fn autodiff_viscous_dissipation(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
                velocities: &Self::Velocities,
            ) -> Quantity<Power> {
                viscous_dissipation::<M, 3, $g, $n, $o, { 3 * $n }, { 3 * $n * $g }>(
                    model,
                    self,
                    coordinates,
                    velocities,
                )
            }
            fn autodiff_helmholtz_free_energy(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Quantity<Energy> {
                helmholtz_free_energy::<M, 3, $g, $n, $o, { 3 * $n }, { 3 * $n * $g }>(
                    model,
                    self,
                    coordinates,
                )
            }
        }
    };
}

shape!(8, 8, 1);
shape!(1, 4, 1);
