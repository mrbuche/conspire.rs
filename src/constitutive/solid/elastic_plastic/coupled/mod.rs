#[cfg(test)]
mod test;

mod condensed;
mod linearization;
mod monolithic;
mod sensitivities;

pub(crate) use condensed::{condensed, solve, updated_state};
#[cfg(feature = "fem")]
pub(crate) use monolithic::{Monolithic, monolithic_evaluate};
pub(crate) use monolithic::{
    monolithic_plastic, monolithic_residual_local, monolithic_state, monolithic_tangents,
};
#[cfg(test)]
use sensitivities::{Iterate, Sensitivities};

use super::ElasticPlastic;
use crate::{
    constitutive::ConstitutiveError,
    math::TensorRank2,
    mechanics::{FlowDirectionPlastic, Scalar},
};
use std::{array::from_fn, fmt::Debug};

pub(crate) const SIZE: usize = 10;

type Unknowns = [Scalar; SIZE];

fn basis<I, J>(a: usize, b: usize) -> TensorRank2<3, I, J> {
    from_fn(|i| from_fn(|j| if i == a && j == b { 1.0 } else { 0.0 })).into()
}

fn failure<C: ElasticPlastic>(model: &C, error: &dyn Debug) -> ConstitutiveError {
    ConstitutiveError::custom(format!("{error:?}"), model)
}

fn increment(x: &Unknowns) -> FlowDirectionPlastic {
    from_fn(|i| from_fn(|j| x[3 * i + j])).into()
}

fn reference<C: ElasticPlastic>(model: &C) -> Scalar {
    let initial = model.initial_yield_stress().value();
    if initial > 0.0 { initial } else { 1.0 }
}
