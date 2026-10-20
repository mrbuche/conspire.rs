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
pub trait Load<X, R> {
    /// Subtracts the external forces at a state from a residual.
    fn subtract_from(&self, state: &X, residual: &mut R) -> Result<(), String>;
}

/// The stiffness of external forces that depend on the state, such as a pressure that follows the
/// surface it acts on, which a tangent of type `J` takes up.
///
/// A load that does not depend on the state has none to add, so the default adds nothing.
pub trait LoadStiffness<X, J> {
    /// Adds the derivative of the residual of the external forces to a tangent.
    fn add_stiffness_to(&self, _state: &X, _tangent: &mut J) -> Result<(), String> {
        Ok(())
    }
}

/// The potential energy of external forces that have one, which may be a function of time.
pub trait ConservativeAt<X, E> {
    /// The potential energy of the external forces at a time and a state.
    fn potential_energy_at(&self, time: Quantity<Time>, state: &X) -> Result<E, String>;
}

/// The potential energy of external forces that have one, so that a minimization can take them up.
pub trait Conservative<X, E>: ConservativeAt<X, E> {
    /// The potential energy of the external forces at a state.
    fn potential_energy(&self, state: &X) -> Result<E, String>;
}

/// The power of external forces, which a minimization over the rates of a state takes up.
///
/// The power is linear in the rates, so loads that are not conservative have one.
pub trait PowerAt<X, V, P> {
    /// The power of the external forces at a time and a state, through rates.
    fn power_at(&self, time: Quantity<Time>, state: &X, rates: &V) -> Result<P, String>;
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

/// The stiffness of external forces that depend on the state and may be a function of time.
///
/// A load that does not depend on the state has none to add, so the default adds nothing.
pub trait LoadStiffnessAt<X, J> {
    /// Adds the derivative of the residual of the external forces at a time to a tangent.
    fn add_stiffness_at(
        &self,
        _time: Quantity<Time>,
        _state: &X,
        _tangent: &mut J,
    ) -> Result<(), String> {
        Ok(())
    }
}

impl<X, J> LoadStiffness<X, J> for () {}

impl<X, J> LoadStiffnessAt<X, J> for () {}

impl<X, J, A, B> LoadStiffnessAt<X, J> for (A, B)
where
    A: LoadStiffnessAt<X, J>,
    B: LoadStiffnessAt<X, J>,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        state: &X,
        tangent: &mut J,
    ) -> Result<(), String> {
        self.0.add_stiffness_at(time, state, tangent)?;
        self.1.add_stiffness_at(time, state, tangent)
    }
}

impl<L, T, X, J> LoadStiffnessAt<X, J> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: LoadStiffnessAt<X, J>,
{
    fn add_stiffness_at(
        &self,
        time: Quantity<Time>,
        state: &X,
        tangent: &mut J,
    ) -> Result<(), String> {
        self.0.add_stiffness_at(time, state, tangent)
    }
}

impl<X, J, A, B> LoadStiffness<X, J> for (A, B)
where
    A: LoadStiffness<X, J>,
    B: LoadStiffness<X, J>,
{
    fn add_stiffness_to(&self, state: &X, tangent: &mut J) -> Result<(), String> {
        self.0.add_stiffness_to(state, tangent)?;
        self.1.add_stiffness_to(state, tangent)
    }
}

impl<L, T, X, J> LoadStiffness<X, J> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: LoadStiffness<X, J>,
{
    fn add_stiffness_to(&self, state: &X, tangent: &mut J) -> Result<(), String> {
        self.0.add_stiffness_to(state, tangent)
    }
}

impl<X, E> ConservativeAt<X, E> for ()
where
    E: Default,
{
    fn potential_energy_at(&self, _: Quantity<Time>, _: &X) -> Result<E, String> {
        Ok(E::default())
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

impl<X, V, P> PowerAt<X, V, P> for ()
where
    P: Default,
{
    fn power_at(&self, _: Quantity<Time>, _: &X, _: &V) -> Result<P, String> {
        Ok(P::default())
    }
}

impl<X, V, P, A, B> PowerAt<X, V, P> for (A, B)
where
    A: PowerAt<X, V, P>,
    B: PowerAt<X, V, P>,
    P: Add<Output = P>,
{
    fn power_at(&self, time: Quantity<Time>, state: &X, rates: &V) -> Result<P, String> {
        Ok(self.0.power_at(time, state, rates)? + self.1.power_at(time, state, rates)?)
    }
}

impl<L, T, X, V, P> PowerAt<X, V, P> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: PowerAt<X, V, P>,
{
    fn power_at(&self, time: Quantity<Time>, state: &X, rates: &V) -> Result<P, String> {
        self.0.power_at(time, state, rates)
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

impl<X, E, A, B> ConservativeAt<X, E> for (A, B)
where
    A: ConservativeAt<X, E>,
    B: ConservativeAt<X, E>,
    E: Add<Output = E>,
{
    fn potential_energy_at(&self, time: Quantity<Time>, state: &X) -> Result<E, String> {
        Ok(self.0.potential_energy_at(time, state)? + self.1.potential_energy_at(time, state)?)
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

impl<L, T, X, E> ConservativeAt<X, E> for Loads<L, T>
where
    X: NodalState<Layout = L>,
    T: ConservativeAt<X, E>,
{
    fn potential_energy_at(&self, time: Quantity<Time>, state: &X) -> Result<E, String> {
        self.0.potential_energy_at(time, state)
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
