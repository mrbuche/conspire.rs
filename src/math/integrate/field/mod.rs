mod adaptive;
mod euler;
mod hermite;
mod rkmk;
mod state;
#[cfg(test)]
mod test;

use crate::math::{
    Tensor, TensorError, TensorRank2, TensorTuple, TensorVector, integrate::IntegrationError,
};
use crate::units::Dimensionless;
use std::{
    marker::PhantomData,
    ops::{Add, Mul},
};

pub use adaptive::{
    integrate_rkmk, integrate_rkmk_adaptive, integrate_rkmk_dae_adaptive,
    integrate_rkmk_dae_adaptive_first_order_root,
    integrate_rkmk_dae_adaptive_second_order_minimize,
};
pub use euler::integrate_euler;
pub use hermite::{HermiteSegment, interpolate_hermite};
pub use rkmk::{
    rkmk_dae_step, rkmk_dae_step_first_order_root, rkmk_dae_step_second_order_minimize, rkmk_step,
};
pub use state::{
    EvolvedIncrement, EvolvedState, StateEvolution, integrate_rkmk_state,
    integrate_rkmk_state_adaptive,
};

const RECONSTRUCT_FAILED: &str =
    "the field increment has no reconstruction (matrix exponential undefined)";

/// The geometry of one integrated state field: how an increment advances the state.
///
/// [`Self::Increment`] is an element of the field's tangent space (its Lie algebra
/// for a group-valued field). It equals [`Self::Point`] for a flat field, but not
/// in general — e.g. `F_p` is a `Reference → Intermediate` map while its algebra
/// element `D_p Δt` maps `Intermediate → Intermediate`.
pub trait Integrable {
    /// The state value this field carries.
    type Point: Tensor;
    /// The tangent/algebra element that advances a [`Self::Point`].
    type Increment: Tensor;
    /// Advances `base` by `increment`.
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError>;
    /// The RKMK correction: maps a rate-scaled increment to an algebra increment
    /// at the accumulated algebra element `sigma`. Flat fields are the identity.
    fn dexpinv(_sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
        increment
    }
}

/// A state in a flat vector space: the increment simply adds.
pub struct Flat<T>(PhantomData<T>);

impl<T> Integrable for Flat<T>
where
    T: Clone + Tensor,
    for<'a> T: Add<&'a T, Output = T>,
{
    type Point = T;
    type Increment = T;
    fn reconstruct(base: &T, increment: &T) -> Result<T, TensorError> {
        Ok(base.clone() + increment)
    }
}

/// A state acted on by the matrix exponential, `X_{n+1} = exp(increment) X_n`,
/// staying on the unimodular group (`det = 1`) whenever the increment is
/// trace-free. The state maps `B → A` while its algebra element maps `A → A`,
/// so `F_p` (`Reference → Intermediate`) is `Unimodular<Intermediate, Reference>`.
pub struct Unimodular<A, B = A>(PhantomData<(A, B)>);

impl<A, B> Integrable for Unimodular<A, B>
where
    TensorRank2<3, A, B, Dimensionless>: Tensor,
    TensorRank2<3, A, A, Dimensionless>: Tensor,
    for<'a> TensorRank2<3, A, A, Dimensionless>:
        Mul<&'a TensorRank2<3, A, B, Dimensionless>, Output = TensorRank2<3, A, B, Dimensionless>>,
{
    type Point = TensorRank2<3, A, B, Dimensionless>;
    type Increment = TensorRank2<3, A, A, Dimensionless>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError> {
        Ok(increment.expm()? * base)
    }
    fn dexpinv(sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
        sigma.dexpinv(&increment)
    }
}

/// A composite of two fields; its state is the matching [`TensorTuple`], and an
/// increment reconstructs component-wise. Nests right for three or more fields.
pub struct Product<H, T>(PhantomData<(H, T)>);

impl<H, T> Integrable for Product<H, T>
where
    H: Integrable,
    T: Integrable,
    TensorTuple<H::Point, T::Point>: Tensor,
    TensorTuple<H::Increment, T::Increment>: Tensor,
{
    type Point = TensorTuple<H::Point, T::Point>;
    type Increment = TensorTuple<H::Increment, T::Increment>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError> {
        Ok(TensorTuple(
            H::reconstruct(&base.0, &increment.0)?,
            T::reconstruct(&base.1, &increment.1)?,
        ))
    }
    fn dexpinv(sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
        TensorTuple(
            H::dexpinv(&sigma.0, increment.0),
            T::dexpinv(&sigma.1, increment.1),
        )
    }
}

/// A list of independent copies of one field, e.g. every Gauss point's plastic
/// state across a mesh; an increment reconstructs entry-wise. Composes with
/// [`Product`] for a multi-block mesh (`Product<List<Fld1>, List<Fld2>>`).
pub struct List<Fld>(PhantomData<Fld>);

impl<Fld> Integrable for List<Fld>
where
    Fld: Integrable,
    TensorVector<Fld::Point>: Tensor<Item = Fld::Point>,
    TensorVector<Fld::Increment>: Tensor<Item = Fld::Increment>,
{
    type Point = TensorVector<Fld::Point>;
    type Increment = TensorVector<Fld::Increment>;
    fn reconstruct(
        base: &Self::Point,
        increment: &Self::Increment,
    ) -> Result<Self::Point, TensorError> {
        base.iter()
            .zip(increment.iter())
            .map(|(base, increment)| Fld::reconstruct(base, increment))
            .collect()
    }
    fn dexpinv(sigma: &Self::Increment, increment: Self::Increment) -> Self::Increment {
        sigma
            .iter()
            .zip(increment)
            .map(|(sigma, increment)| Fld::dexpinv(sigma, increment))
            .collect()
    }
}

fn reconstruct_or_err<Field: Integrable>(
    base: &Field::Point,
    increment: &Field::Increment,
) -> Result<Field::Point, IntegrationError> {
    Field::reconstruct(base, increment)
        .map_err(|_| IntegrationError::from(RECONSTRUCT_FAILED.to_string()))
}
