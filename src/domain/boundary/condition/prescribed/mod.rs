#[cfg(test)]
mod test;

use super::{Condition, Enforcement, Static, Target, held};
use crate::{
    domain::boundary::{Component, Layout},
    math::{
        Quantity,
        optimize::{EqualityConstraint, Prescription},
    },
    units::Time,
};
use std::marker::PhantomData;

/// A component of some nodes held at a value.
///
/// The nodes are a set that the condition borrows, or are listed.
pub struct Prescribed<'m, L> {
    target: Target<'m>,
    stride: usize,
    offset: usize,
    prescription: Prescription,
    enforcement: Enforcement,
    layout: PhantomData<L>,
}

impl<'m, L> Prescribed<'m, L>
where
    L: Layout,
{
    pub(super) fn new<C, I>(
        target: Target<'m>,
        prescription: Prescription,
        enforcement: Enforcement,
    ) -> Self
    where
        C: Component<L, I>,
    {
        Self {
            target,
            stride: L::STRIDE,
            offset: <C as Component<L, I>>::OFFSET,
            prescription,
            enforcement,
            layout: PhantomData,
        }
    }
}

impl<L> Prescribed<'_, L> {
    /// The degrees of freedom held.
    pub fn indices(&self) -> Vec<usize> {
        self.target.indices(self.stride, self.offset)
    }
    /// What the degrees of freedom are held at.
    pub fn prescription(&self) -> Prescription {
        self.prescription
    }
    /// How the degrees of freedom are held.
    pub fn enforcement(&self) -> Enforcement {
        self.enforcement
    }
}

impl<L> Condition<L> for Prescribed<'_, L> {
    fn constraint_at(&self, _: Quantity<Time>) -> EqualityConstraint {
        Static::held_constraint(self)
    }
}

impl<L> Static<L> for Prescribed<'_, L> {
    fn held_constraint(&self) -> EqualityConstraint {
        held(
            self.indices()
                .into_iter()
                .map(|index| (index, self.prescription))
                .collect(),
            self.enforcement,
        )
    }
}
