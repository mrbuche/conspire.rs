#[cfg(test)]
mod test;

use super::{Condition, Enforcement, Target, held};
use crate::{
    domain::boundary::{Component, Layout, Prescribable},
    math::{
        Quantity,
        optimize::{EqualityConstraint, Prescription},
    },
    units::Time,
};
use std::marker::PhantomData;

/// A component of some nodes held at a value that is a function of time, by elimination.
pub struct Timed<'m, L, G> {
    target: Target<'m>,
    stride: usize,
    offset: usize,
    prescription: G,
    layout: PhantomData<L>,
}

impl<'m, L> Timed<'m, L, ()>
where
    L: Layout,
{
    pub(super) fn new<C, I>(
        target: Target<'m>,
        _: C,
        value: impl Fn(Quantity<Time>) -> C::Value,
    ) -> Timed<'m, L, impl Fn(Quantity<Time>) -> Prescription>
    where
        C: Component<L, I> + Prescribable,
    {
        Timed {
            target,
            stride: L::STRIDE,
            offset: <C as Component<L, I>>::OFFSET,
            prescription: move |time| C::prescription(value(time)),
            layout: PhantomData,
        }
    }
}

impl<L, G> Condition<L> for Timed<'_, L, G>
where
    G: Fn(Quantity<Time>) -> Prescription,
{
    fn constraint_at(&self, time: Quantity<Time>) -> EqualityConstraint {
        let prescription = (self.prescription)(time);
        held(
            self.target
                .indices(self.stride, self.offset)
                .into_iter()
                .map(|index| (index, prescription))
                .collect(),
            Enforcement::Elimination,
        )
    }
}
