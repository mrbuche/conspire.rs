#![allow(clippy::needless_range_loop)]

use crate::{
    constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
    fem::block::element::{
        Element, FiniteElement,
        autodiff::{Coordinates, flatten, unflatten},
        solid::autodiff::{
            Forces, Stiffnesses, assemble_tangent, deformation_gradient, tangent_matrix,
        },
    },
};
use std::autodiff::autodiff_reverse;

#[autodiff_reverse(d_element_energy, Const, Const, Const, Duplicated, Active)]
pub(crate) fn element_energy<
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

pub(crate) fn forces_flat<
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

pub(crate) fn forces<
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
    unflatten::<D, N, DOF>(&flat).into()
}

pub(crate) fn stiffnesses<
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
    let parameters = model.parameters();
    let (grad_n, weights, x) = flatten::<D, G, N, O, DOF, GN>(element, coordinates);
    let mut tangents = [[0.0; 81]; G];
    for g in 0..G {
        let f = deformation_gradient::<D, N, DOF, GN>(&grad_n, g, &x);
        tangents[g] = tangent_matrix(|direction, primal, seed| {
            M::piola_tangent(&parameters, &f, direction, primal, seed)
        });
    }
    assemble_tangent::<D, N, G, GN>(&tangents, &grad_n, &weights).into()
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

macro_rules! autodiff_element {
    ($d:literal, $g:literal, $n:literal, $o:literal) => {
        impl<M> $crate::fem::block::element::solid::hyperelastic::autodiff::AutodiffElement<M>
            for $crate::fem::block::element::Element<$d, $g, $n, $o>
        where
            M: $crate::constitutive::solid::hyperelastic::autodiff::AutodiffHyperelastic,
        {
            type Coordinates = $crate::fem::block::element::autodiff::Coordinates<$d, $n>;
            type Forces = $crate::fem::block::element::solid::autodiff::Forces<$d, $n>;
            type Stiffnesses = $crate::fem::block::element::solid::autodiff::Stiffnesses<$d, $n>;
            fn autodiff_nodal_forces(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Self::Forces {
                $crate::fem::block::element::solid::hyperelastic::autodiff::forces::<
                    M,
                    $d,
                    $g,
                    $n,
                    $o,
                    { $d * $n },
                    { $d * $n * $g },
                >(model, self, coordinates)
            }
            fn autodiff_nodal_stiffnesses(
                &self,
                model: &M,
                coordinates: &Self::Coordinates,
            ) -> Self::Stiffnesses {
                $crate::fem::block::element::solid::hyperelastic::autodiff::stiffnesses::<
                    M,
                    $d,
                    $g,
                    $n,
                    $o,
                    { $d * $n },
                    { $d * $n * $g },
                >(model, self, coordinates)
            }
        }
    };
}

pub(crate) use autodiff_element;
