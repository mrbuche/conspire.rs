#[cfg(test)]
mod test;

use crate::{domain::boundary::NodalState, math::Quantity, units::Time};
use std::{marker::PhantomData, ops::Add};

/// External forces on the degrees of freedom of a state, which may be a function of time.
pub trait LoadAt<X, R> {
    /// Subtracts the external forces at a time and a state from a residual.
    fn subtract_at(&self, time: Quantity<Time>, state: &X, residual: &mut R) -> Result<(), String>;
}

/// External forces on the degrees of freedom of a state, which a residual of type `R` takes up.
///
/// A solve that is not a function of time takes these, so loads that are a function of time are
/// not accepted by it.
pub trait Load<X, R>: LoadAt<X, R> {
    /// Subtracts the external forces at a state from a residual.
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String>;
}

/// The potential energy of external forces that have one, so that a minimization can take them up.
pub trait Conservative<X, E> {
    /// The potential energy of the external forces at a state.
    fn potential_energy(&self, state: &X) -> Result<E, String>;
}

/// External loads on the degrees of freedom of a layout, held in a nest of pairs.
///
/// The loads a layout can have are the builders defined for it, so that a load cannot be given
/// to a solve of another layout.
#[derive(Clone, Debug)]
pub struct Loads<L, T = ()>(T, PhantomData<L>);

impl<L> Loads<L> {
    pub fn new() -> Self {
        Self((), PhantomData)
    }
}

impl<L> Default for Loads<L> {
    fn default() -> Self {
        Self::new()
    }
}

impl<L, T> Loads<L, T> {
    /// Adds a load.
    pub fn with<A>(self, load: A) -> Loads<L, (T, A)> {
        Loads((self.0, load), PhantomData)
    }
}

impl<X, R> LoadAt<X, R> for () {
    fn subtract_at(&self, _: Quantity<Time>, _: &X, _: &mut R) -> Result<(), String> {
        Ok(())
    }
}

impl<X, R> Load<X, R> for () {
    fn subtract_from(&self, _: &X, _: &mut R) -> Result<(), String> {
        Ok(())
    }
}

impl<X, E> Conservative<X, E> for ()
where
    E: Default,
{
    fn potential_energy(&self, _: &X) -> Result<E, String> {
        Ok(E::default())
    }
}

impl<X, R, A, B> LoadAt<X, R> for (A, B)
where
    A: LoadAt<X, R>,
    B: LoadAt<X, R>,
{
    fn subtract_at(&self, time: Quantity<Time>, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_at(time, state, residual)?;
        self.1.subtract_at(time, state, residual)
    }
}

impl<X, R, A, B> Load<X, R> for (A, B)
where
    A: Load<X, R>,
    B: Load<X, R>,
{
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_from(state, residual)?;
        self.1.subtract_from(state, residual)
    }
}

impl<X, E, A, B> Conservative<X, E> for (A, B)
where
    A: Conservative<X, E>,
    B: Conservative<X, E>,
    E: Add<Output = E>,
{
    fn potential_energy(&self, state: &X) -> Result<E, String> {
        Ok(self.0.potential_energy(state)? + self.1.potential_energy(state)?)
    }
}

impl<L, T, X, R> LoadAt<X, R> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: LoadAt<X, R>,
{
    fn subtract_at(&self, time: Quantity<Time>, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_at(time, state, residual)
    }
}

impl<L, T, X, R> Load<X, R> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: Load<X, R>,
{
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String> {
        self.0.subtract_from(state, residual)
    }
}

impl<L, T, X, E> Conservative<X, E> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: Conservative<X, E>,
{
    fn potential_energy(&self, state: &X) -> Result<E, String> {
        self.0.potential_energy(state)
    }
}
