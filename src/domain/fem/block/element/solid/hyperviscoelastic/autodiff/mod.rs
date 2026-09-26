#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::{
        fluid::{hyperviscous::autodiff::AutodiffHyperviscous, viscous::autodiff::AutodiffViscous},
        solid::hyperviscoelastic::autodiff::AutodiffHyperviscoelastic,
    },
    fem::block::element::{
        Element, FiniteElement,
        autodiff::{Coordinates, flatten, unflatten},
        solid::{
            autodiff::{
                Dampings, Forces, Velocities, assemble_tangent, deformation_gradient,
                deformation_gradient_rate, flatten_velocities, tangent_matrix,
            },
            hyperelastic::autodiff::{element_energy, forces_flat},
        },
    },
    math::Quantity,
    units::{Energy, Power},
};
use std::autodiff::autodiff_reverse;

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

pub(crate) fn forces<
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
    let mut total = [0.0; DOF];
    for k in 0..DOF {
        total[k] = elastic[k] + viscous[k];
    }
    unflatten::<D, N, DOF>(&total).into()
}

pub(crate) fn dampings<
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
    let parameters = model.viscous_parameters();
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let v = flatten_velocities::<D, N, DOF>(velocities);
    let mut tangents = [[0.0; 81]; G];
    for g in 0..G {
        let f = deformation_gradient::<D, N, DOF, GN>(&grad_n, g, &x);
        let f_dot = deformation_gradient_rate::<D, N, DOF, GN>(&grad_n, g, &v);
        tangents[g] = tangent_matrix(|direction, primal, seed| {
            M::Viscous::viscous_piola_tangent(&parameters, &f, &f_dot, direction, primal, seed)
        });
    }
    assemble_tangent::<D, N, G, GN>(&tangents, &grad_n, &weights).into()
}

pub(crate) fn viscous_dissipation<
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

pub(crate) fn helmholtz_free_energy<
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

macro_rules! autodiff_viscoelastic_element {
    ($g:literal, $n:literal, $o:literal) => {
        impl<M> $crate::fem::block::element::solid::hyperviscoelastic::autodiff::AutodiffViscoelasticElement<M>
            for $crate::fem::block::element::Element<3, $g, $n, $o>
        where
            M: $crate::constitutive::solid::hyperviscoelastic::autodiff::AutodiffHyperviscoelastic,
        {
            type Coordinates = $crate::fem::block::element::ElementNodalCoordinates<$n>;
            type Velocities = $crate::fem::block::element::ElementNodalVelocities<$n>;
            type Forces =
                $crate::fem::block::element::solid::autodiff::Forces<3, $n>;
            type Dampings = $crate::fem::block::element::solid::ElementNodalDampingsSolid<$n>;
            fn autodiff_viscoelastic_nodal_forces(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
                velocities: &Self::Velocities,
            ) -> Self::Forces {
                $crate::fem::block::element::solid::hyperviscoelastic::autodiff::forces::<
                    M,
                    3,
                    $g,
                    $n,
                    $o,
                    { 3 * $n },
                    { 3 * $n * $g },
                >(model, self, coordinates, velocities)
            }
            fn autodiff_nodal_dampings(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
                velocities: &Self::Velocities,
            ) -> Self::Dampings {
                $crate::fem::block::element::solid::hyperviscoelastic::autodiff::dampings::<
                    M,
                    3,
                    $g,
                    $n,
                    $o,
                    { 3 * $n },
                    { 3 * $n * $g },
                >(model, self, coordinates, velocities)
            }
            fn autodiff_viscous_dissipation(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
                velocities: &Self::Velocities,
            ) -> $crate::math::Quantity<$crate::units::Power> {
                $crate::fem::block::element::solid::hyperviscoelastic::autodiff::viscous_dissipation::<
                    M,
                    3,
                    $g,
                    $n,
                    $o,
                    { 3 * $n },
                    { 3 * $n * $g },
                >(model, self, coordinates, velocities)
            }
            fn autodiff_helmholtz_free_energy(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> $crate::math::Quantity<$crate::units::Energy> {
                $crate::fem::block::element::solid::hyperviscoelastic::autodiff::helmholtz_free_energy::<
                    M,
                    3,
                    $g,
                    $n,
                    $o,
                    { 3 * $n },
                    { 3 * $n * $g },
                >(model, self, coordinates)
            }
        }
    };
}

pub(crate) use autodiff_viscoelastic_element;
