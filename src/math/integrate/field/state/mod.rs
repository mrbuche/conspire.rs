use super::{
    Integrable,
    adaptive::{integrate_rkmk, integrate_rkmk_adaptive},
};
use crate::math::{
    Derivative, Differentiable, Quantity, Scalar, TensorVec,
    integrate::{ButcherTableau, EmbeddedTableau, IntegrationError, Times},
};
use crate::units::Time;
use std::ops::Mul;

/// The `Point` type of a [`StateEvolution`] model's field.
pub type EvolvedState<M, T = Time, Y = Quantity> =
    <<M as StateEvolution<T, Y>>::Field as Integrable>::Point;

/// The `Increment` (Lie-algebra) type of a [`StateEvolution`] model's field.
pub type EvolvedIncrement<M, T = Time, Y = Quantity> =
    <<M as StateEvolution<T, Y>>::Field as Integrable>::Increment;

/// A model whose internal state evolves as a product of Lie-algebra rates,
/// ready for the field drivers. [`Self::Drive`] is the externally-imposed input
/// the rate needs beside the state (e.g. the total deformation gradient).
///
/// `Y` is only a discriminant: a model type (e.g. `Canonical`) that could carry
/// several kinds of internal state selects one impl per `Y`, so it appears here
/// even though nothing in the trait names it.
pub trait StateEvolution<T = Time, Y = Quantity>
where
    <Self::Field as Integrable>::Increment: Differentiable<T>,
{
    /// Geometry of the composite internal state.
    type Field: Integrable;
    /// The externally-imposed driving input.
    type Drive;
    /// The initial internal state.
    fn initial_state(&self) -> <Self::Field as Integrable>::Point;
    /// The product of Lie-algebra rates at `(time, drive, state)`.
    fn state_rate(
        &self,
        time: Quantity<T>,
        drive: &Self::Drive,
        state: &<Self::Field as Integrable>::Point,
    ) -> Result<Derivative<<Self::Field as Integrable>::Increment, T>, String>;
}

/// Runs [`super::integrate_rkmk`] over a [`StateEvolution`] model, sampling
/// `drive` at each stage time and starting from the model's own initial state.
pub fn integrate_rkmk_state<M, Tab, U, T, Y>(
    model: &M,
    mut drive: impl FnMut(Quantity<T>) -> M::Drive,
    time: &[Quantity<T>],
) -> Result<(Times<T>, U), IntegrationError>
where
    M: StateEvolution<T, Y>,
    Tab: ButcherTableau,
    EvolvedState<M, T, Y>: Clone,
    EvolvedIncrement<M, T, Y>: Clone + Differentiable<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<EvolvedIncrement<M, T, Y>, T>:
        Mul<Quantity<T>, Output = EvolvedIncrement<M, T, Y>>,
    U: TensorVec<Item = EvolvedState<M, T, Y>>,
{
    let initial = model.initial_state();
    integrate_rkmk::<M::Field, Tab, U, T>(
        |t, state| model.state_rate(t, &drive(t), state),
        time,
        initial,
    )
}

/// Runs [`super::integrate_rkmk_adaptive`] over a [`StateEvolution`] model,
/// sampling `drive` at each stage time and starting from the model's own
/// initial state. `time` supplies only the span `[time[0], time[last]]`; the
/// returned times are the steps the controller accepted.
pub fn integrate_rkmk_state_adaptive<M, Tab, U, T, Y>(
    model: &M,
    mut drive: impl FnMut(Quantity<T>) -> M::Drive,
    time: &[Quantity<T>],
    abs_tol: Scalar,
    rel_tol: Scalar,
) -> Result<(Times<T>, U), IntegrationError>
where
    M: StateEvolution<T, Y>,
    Tab: EmbeddedTableau,
    EvolvedState<M, T, Y>: Clone,
    EvolvedIncrement<M, T, Y>: Clone + Differentiable<T>,
    T: Copy,
    Quantity<T>: Mul<Scalar, Output = Quantity<T>>,
    for<'a> &'a Derivative<EvolvedIncrement<M, T, Y>, T>:
        Mul<Quantity<T>, Output = EvolvedIncrement<M, T, Y>>,
    U: TensorVec<Item = EvolvedState<M, T, Y>>,
{
    let initial = model.initial_state();
    integrate_rkmk_adaptive::<M::Field, Tab, U, T>(
        |t, state| model.state_rate(t, &drive(t), state),
        time,
        initial,
        abs_tol,
        rel_tol,
    )
}

//
// `StateStep` used to sit here: a seam meant to let a group-valued state
// override the additive Runge–Kutta march. It could never work. Its slope was
// typed `Derivative<Self, T>`, and `Differentiable` admits exactly one
// `Derivative` per state — for `(F_p, Y)` that is the group velocity `Ḟ_p`
// (`Intermediate ← Reference`), while RKMK needs the algebra element `D_p`
// (`Intermediate ← Intermediate`) for the same state. No impl can supply a
// second slope type, so the manifold branch the trait advertised was
// unreachable (the coherence error it surfaced as was only a symptom).
//
// Manifold stepping instead dispatches on the field — `Integrable`, whose
// `Point`/`Increment` split carries exactly that distinction and which no state
// type can collide with. The additive march is now inline in the two
// Runge–Kutta loops that used the trait.
//
